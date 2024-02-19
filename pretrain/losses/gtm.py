import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
from tqdm import tqdm
from einops import rearrange, repeat

from model.base import *
from model.layers import get_batch_mask, gen_casual_mask, MLP, ContinuousEncoding, MultiHeadAttention, Unsqueeze
from data import Denormalizer


def valid_flat(*xs, length):
    return (pack_padded_sequence(x, length.long().cpu(), batch_first=True, enforce_sorted=False).data for x in xs)


def flat_seq_i(seq_i, L):
    B = seq_i.size(0)
    seq_i = seq_i + torch.arange(B).to(seq_i.device).unsqueeze(-1) * L
    seq_i = seq_i.reshape(-1)
    return seq_i


def gen_transformer_mask(L_dual, L_reg, dual_valid_len, reg_valid_len):
    B = dual_valid_len.size(0)
    # The batch mask is a concatenation of valid trip steps and valid source steps.
    batch_mask = torch.cat([get_batch_mask(B, L_dual, dual_valid_len),
                            get_batch_mask(B, L_reg, reg_valid_len)], 1)  # (B, L)

    L = L_dual + L_reg
    source_mask = torch.cat([torch.zeros(L, L_dual), torch.ones(L, L_reg)], 1).to(dual_valid_len.device).bool()
    source_mask = source_mask & gen_casual_mask(L).to(dual_valid_len.device).bool()

    return batch_mask, source_mask


class GTM(Loss):
    """
    General Trajectory Model.
    """

    def __init__(self, num_tokens, num_class_dict, latent_size, cand_arg_dict, extra_params,
                 spatial_weight=1.0, temporal_weight=1.0, roadnet_weight=1.0,
                 model_name_suf='', ablation='none'):
        """
        :param num_tokens: number of road network tokens.
        :param num_class_dict: dictionary containing the number of classes for various discrete features.
        :param latent_size: latent size of embeddings and linear layers.
        :param cand_arg_dict: dictionary containing the argument indices for various candidate meta data.
        """
        super().__init__(
            f'GTM-latent{latent_size}-spa{spatial_weight}-temp{temporal_weight}-rn{roadnet_weight}-{model_name_suf}' +
            ('' if ablation == 'none' else f'-AB{ablation}'))

        self.ablation = ablation  # wo_neigh, wo_coor, wo_time, fc_numenc
        self.num_tokens = num_tokens
        self.num_class_dict = num_class_dict
        self.latent_size = latent_size
        self.cand_arg_dict = cand_arg_dict

        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.roadnet_weight = roadnet_weight

        self.num_special_tokens = 4
        self.num_feat_types = 3
        self.num_expanded_tokens = num_tokens + self.num_feat_types * self.num_special_tokens

        self.extra_params = nn.ParameterList([nn.Parameter(torch.from_numpy(param).float()) for param in extra_params])

        self.in_cand_embed = nn.Embedding(num_tokens + 1, latent_size, padding_idx=num_tokens)
        self.token_embed = nn.Embedding(self.num_expanded_tokens + 1, latent_size, padding_idx=self.num_expanded_tokens)
        self.dis_embeds = nn.ModuleDict({key: nn.Embedding(num_class + 1, latent_size, padding_idx=num_class)
                                        for key, num_class in num_class_dict.items()})
        if 'fc_numenc' in self.ablation:
            self.con_embeds = nn.ModuleDict({key: nn.Sequential(Unsqueeze(-1), nn.Linear(1, latent_size))
                                             for key in ['lng', 'lat', 'tod', 'roadprop']})
        else:
            self.con_embeds = nn.ModuleDict({key: ContinuousEncoding(latent_size)
                                             for key in ['lng', 'lat', 'tod', 'roadprop']})

        self.in_seq_cand_att = MultiHeadAttention(latent_size, num_heads=8, dropout=0.1)
        self.in_token_cand_att = MultiHeadAttention(latent_size, num_heads=8, dropout=0.1)

        self.road_token_predictor = MLP(latent_size, latent_size * 4, num_tokens + self.num_special_tokens)
        self.coordinate_predictor = nn.Sequential(MLP(latent_size, latent_size // 4, 2), nn.Tanh())
        self.roadprop_predictor = nn.Sequential(MLP(latent_size, latent_size // 4, 1), nn.Tanh())
        self.offset_predictor = nn.Sequential(MLP(latent_size, latent_size // 4, 1), nn.Softplus())

    def _cal_latent(self, seq_i, tokens, weekday, coordinate, tod, roadprop=None, in_seq_cand=None):
        """
        Calculate the input latent sequence given raw features.

        :param tokens: tokens corresponding to multiple features, with shape (B, L, N_type).
            There should be at least three types of features: spatial, temporal, and roadnet.
        """
        B = tokens.size(0)

        expanded_tokens = tokens + repeat(torch.arange(self.num_feat_types).to(tokens.device)
                                          * self.num_special_tokens, 'N -> 1 1 N')
        expanded_tokens = torch.where(tokens >= self.num_tokens, expanded_tokens, tokens)
        expanded_tokens = expanded_tokens.masked_fill(expanded_tokens < 0, self.num_expanded_tokens).long()
        token_latent = self.token_embed(expanded_tokens)  # (B, L, N_type, E)

        weekday = weekday.masked_fill(weekday < 0, self.num_class_dict['weekday']).long()
        weekday_latent = self.dis_embeds['weekday'](weekday)  # (B, L, E)

        coordinate_latent = torch.stack([self.con_embeds[key](coordinate[..., i])
                                         for i, key in enumerate(['lng', 'lat'])], -1).sum(-1)  # (B, L, E)
        coordinate_latent = coordinate_latent.masked_fill((coordinate[..., 0] < -100).unsqueeze(-1), 0)

        tod_latent = self.con_embeds['tod'](tod)  # (B, L, E)
        tod_latent = tod_latent.masked_fill((tod < -100).unsqueeze(-1), 0)

        spatial_latent = token_latent[:, :, 0]
        if 'wo_coor' not in self.ablation:
            spatial_latent = spatial_latent + coordinate_latent
        temporal_latent = token_latent[:, :, 1] + weekday_latent
        if 'wo_time' not in self.ablation:
            temporal_latent = temporal_latent + tod_latent
        roadnet_latent = token_latent[:, :, 2]

        if roadprop is not None:
            roadprop_latent = self.con_embeds['roadprop'](roadprop)
            roadprop_latent = roadprop_latent.masked_fill((roadprop < -100).unsqueeze(-1), 0)
            roadnet_latent = roadnet_latent + roadprop_latent

        if in_seq_cand is not None and 'wo_neigh' not in self.ablation:
            seq_invalid_mask = seq_i < 0  # (B, L)
            seq_cand = in_seq_cand.masked_fill(in_seq_cand < 0, self.num_tokens).long()
            seq_cand_embeds = self.in_cand_embed(seq_cand)  # (B, L, N, E)
            flat_trip_seq_i = flat_seq_i(seq_i, seq_cand.size(1)).masked_fill(
                seq_invalid_mask.reshape(-1), 0).long()
            num_valid_cand = rearrange(rearrange((in_seq_cand >= 0).long().sum(-1),
                                       'B L -> (B L)')[flat_trip_seq_i], '(B L) -> B L', B=B)

            seq_cand_embeds = rearrange(rearrange(seq_cand_embeds, 'B L N E -> (B L) N E')[flat_trip_seq_i],
                                        'B N E -> N B E')
            cand_invalid_mask = rearrange(in_seq_cand < 0, 'B L N -> (B L) N')[flat_trip_seq_i]
            seq_cand_latent, _ = self.in_seq_cand_att(rearrange(spatial_latent, 'B L E -> 1 (B L) E'),
                                                      seq_cand_embeds, seq_cand_embeds,
                                                      key_padding_mask=cand_invalid_mask)
            seq_cand_latent = rearrange(seq_cand_latent.squeeze(0), '(B L) E -> B L E', B=B)
            seq_cand_latent = seq_cand_latent.masked_fill(seq_invalid_mask.unsqueeze(-1), 0).masked_fill(
                (num_valid_cand < 1).unsqueeze(-1), 0)

            roadnet_latent = roadnet_latent + seq_cand_latent

        latent_list = [spatial_latent, temporal_latent, roadnet_latent]
        latent = torch.stack(latent_list, 2)  # (B, L, N_feat, E)
        return latent

    def _fetch_batch_feats(self, batch):
        seq_i = batch[..., -3]
        tokens = batch[..., :3]
        weekday = batch[..., 3]
        coordinate = batch[..., [5, 6]]
        tod = batch[..., 4]
        return seq_i, tokens, weekday, coordinate, tod

    def _predict(self, latent):
        spatial_latent = latent[:, :, 0]  # (B, L, E)
        coordinate_pred = self.coordinate_predictor(spatial_latent)  # (B, L, 2)
        temporal_latent = latent[:, :, 1]
        offset_pred = self.offset_predictor(temporal_latent).squeeze(-1)  # (B, L)
        roadnet_latent = latent[:, :, 2]
        roadprop_pred = self.roadprop_predictor(roadnet_latent).squeeze(-1)  # (B, L)
        road_token_pred = self.road_token_predictor(roadnet_latent)  # (B, L, N_token)
        return road_token_pred, coordinate_pred, roadprop_pred, offset_pred

    def forward(self, models, enc_metas, rec_metas, con_metas):
        seq_model, = models
        losses = []
        for meta_i in range(len(enc_metas) // 5):
            trip_batch, source_batch, target_batch, trip_len, source_len = enc_metas[meta_i * 5:(meta_i + 1) * 5]
            trip_latent = self._cal_latent(*self._fetch_batch_feats(trip_batch),
                                           in_seq_cand=torch.clone(con_metas[self.cand_arg_dict['in_seq']]))
            trip_positions = trip_batch[..., -2:]
            source_latent = self._cal_latent(*self._fetch_batch_feats(source_batch),
                                             roadprop=source_batch[..., 7])
            source_positions = source_batch[..., -2:]
            latent = torch.cat([trip_latent, source_latent], 1)  # (B, L, N, E)
            positions = torch.cat([trip_positions, source_positions], 1)  # (B, L, 2)

            L_trip, L_source = trip_batch.size(1), source_batch.size(1)
            batch_mask, source_mask = gen_transformer_mask(L_trip, L_source, trip_len, source_len)

            latent = seq_model(latent, positions, src_mask=source_mask, batch_mask=batch_mask)
            rec_latent = latent[:, -L_source:]  # (B, L_src, N, E)

            road_token_pred, coordinate_pred, roadprop_pred, offset_pred = self._predict(rec_latent)
            road_token_true = target_batch[..., 2].long()  # (B, L)
            end_mask = target_batch[..., :3].long() == self.num_tokens + 2

            # Loss on roadnet token
            road_token_true, road_token_pred = valid_flat(road_token_true, road_token_pred, length=source_len)
            road_token_loss = F.cross_entropy(road_token_pred, road_token_true)
            # Loss on coordinates
            coordinate_true = target_batch[..., [3, 4]]  # (B, L, 2)
            coordinate_true, coordinate_pred = (item.masked_fill(end_mask[..., 0].unsqueeze(-1), 0)
                                                for item in [coordinate_true, coordinate_pred])
            coordinate_true, coordinate_pred = valid_flat(coordinate_true, coordinate_pred, length=source_len)
            spatial_loss = F.mse_loss(coordinate_pred, coordinate_true)
            # Loss on roadprop
            roadprop_true = target_batch[..., 5]  # (B, L)
            roadprop_true, roadprop_pred = (item.masked_fill(end_mask[..., 2], 0)
                                            for item in [roadprop_true, roadprop_pred])
            roadprop_true, roadprop_pred = valid_flat(roadprop_true, roadprop_pred, length=source_len)
            roadprop_loss = F.mse_loss(roadprop_pred, roadprop_true)
            # Loss on offset
            offset_true = target_batch[..., 6]  # (B, L)
            offset_true, offset_pred = (item.masked_fill(end_mask[..., 1], 0)
                                        for item in [offset_true, offset_pred])
            offset_true, offset_pred = valid_flat(offset_true, offset_pred, length=source_len)
            offset_loss = F.mse_loss(offset_pred, offset_true)

            loss = self.roadnet_weight * (road_token_loss + roadprop_loss) + \
                self.spatial_weight * spatial_loss + self.temporal_weight * offset_loss
            losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    @torch.no_grad()
    def generate(self, models, enc_metas, rec_metas, con_metas, **gen_params):
        seq_model, = models
        trip_batch, source_batch, target_batch, trip_len, source_len = rec_metas
        trip_positions = trip_batch[..., -2:]
        trip_latent = self._cal_latent(*self._fetch_batch_feats(trip_batch),
                                       in_seq_cand=torch.clone(con_metas[self.cand_arg_dict['in_seq']]))
        L_trip, L_tgt = trip_batch.size(1), target_batch.size(1)

        road_token_arr, coordinate_arr, roadprop_arr, offset_arr = ([] for _ in range(4))
        token_step = source_batch[:, 0:1, :3]  # (B, 1, 3)
        coordinate_step = source_batch[:, 0:1, [5, 6]]  # (B, 1, 2)
        roadprop_step = source_batch[:, 0:1, 7]  # (B, 1)
        latent_till = trip_latent
        for l in range(L_tgt):
            source_step = source_batch[:, l:l+1]  # (B, 1, E)
            seq_i_step, weekday_step, tod_step = source_step[..., -3], \
                source_step[..., 3], source_step[..., 4]  # (B, 1)
            step_latent = self._cal_latent(seq_i_step, token_step, weekday_step, coordinate_step, tod_step,
                                           roadprop=roadprop_step)  # (B, 1, N, E)

            latent_till = torch.cat([latent_till, step_latent], 1)  # (B, L_till, N, E)
            positions_till = torch.cat([trip_positions, source_batch[:, :l+1, -2:]], 1)  # (B, L_till, 2)
            batch_mask_till, source_mask_till = gen_transformer_mask(L_trip, l+1, trip_len, source_len)

            latent = seq_model(latent_till, positions_till, src_mask=source_mask_till,
                               batch_mask=batch_mask_till)  # (B, L_till, N, E)
            latent_next = latent[:, -1:]  # (B, 1, N, E)

            road_token_pred, coordinate_pred, roadprop_pred, offset_pred = self._predict(latent_next)
            road_token_pred = road_token_pred.argmax(-1)
            road_token_arr.append(road_token_pred)
            coordinate_arr.append(coordinate_pred)
            roadprop_arr.append(roadprop_pred)
            offset_arr.append(offset_pred)

            if l < L_tgt - 1:
                source_next = source_batch[:, l+1:l+2]  # (B, 1, E)
                start_mask = source_next[..., :3].long() == self.num_tokens + 1
                token_step = torch.cat([source_next[..., :2],
                                        torch.where(start_mask[..., 2], source_next[..., 2], road_token_pred).unsqueeze(-1)], -1)
                coordinate_step = torch.where(start_mask[..., 0].unsqueeze(-1),
                                              source_next[..., [5, 6]], coordinate_pred)
                roadprop_step = torch.where(start_mask[..., 2], source_next[..., 7], roadprop_pred)

        road_token_arr, coordinate_arr, roadprop_arr, offset_arr = (
            torch.cat(arr, 1) for arr in [road_token_arr, coordinate_arr, roadprop_arr, offset_arr])
        coordinate_denormalizer = Denormalizer(gen_params['stat'], [0, 1], ['lng', 'lat'], 'minmax')
        roadprop_denormalizer = Denormalizer(gen_params['stat'], [0], ['road_prop'], 'minmax')

        gen_dict = {'road_token': road_token_arr.cpu().numpy().astype(int),
                    'coordinate': coordinate_denormalizer([0, 1], coordinate_arr).cpu().numpy(),
                    'roadprop': roadprop_denormalizer([0], roadprop_arr.unsqueeze(-1)).squeeze(-1).cpu().numpy(),
                    'offset': offset_arr.cpu().numpy()}

        return gen_dict, f'GTM-{gen_params["gen_type"]}'
