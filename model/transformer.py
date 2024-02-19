import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

from model.base import *
from model.layers import PositionalEmbedding, MLP, get_batch_mask, gen_casual_mask


class DualPosTransformer(Encoder):
    def __init__(self, d_model, output_size, num_heads=8, num_layers=2, hidden_size=128,
                 mt_strategy='flat', repeat_output=False, **kwargs):
        super().__init__('DualPosTransformer' +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}-o{output_size}' +
                         f'-mt{mt_strategy}-ro{int(repeat_output)}')

        self.mt_strategy = mt_strategy
        self.repeat_output = repeat_output
        if mt_strategy == 'fc':
            self.token_fc = nn.Sequential(nn.Linear(kwargs['input_size'], d_model, bias=False),
                                          nn.LayerNorm(d_model),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Linear(d_model, d_model))
        elif mt_strategy == 'attention':
            token_att_layer = nn.TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
            self.token_att = nn.TransformerEncoder(token_att_layer, num_layers=kwargs['num_token_att_layers'])

        transformer_layer = nn.TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)

        self.pos_embeds = nn.ModuleList([PositionalEmbedding(d_model) for _ in range(2)])
        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

    def forward(self, trip, pos, src_mask=None, batch_mask=None):
        multi_type = len(trip.shape) > 3

        if multi_type:
            B, N_type = trip.size(0), trip.size(2)
            if self.mt_strategy == 'flat':
                trip = rearrange(trip, 'B L N E -> B (L N) E')
                pos = repeat(pos, 'B L E -> B (L N) E', N=N_type)
                src_mask = repeat(src_mask, 'L1 L2 -> (L1 N1) (L2 N2)', N1=N_type, N2=N_type)
                batch_mask = repeat(batch_mask, 'B L -> B (L N)', N=N_type)
            elif self.mt_strategy == 'fc':
                trip = rearrange(trip, 'B L N E -> B L (N E)')
                trip = self.token_fc(trip)
            elif self.mt_strategy == 'mean':
                trip = trip.mean(2)
            elif self.mt_strategy == 'attention':
                trip = rearrange(trip, 'B L N E -> (B L) N E')
                trip = self.token_att(trip)
                trip = rearrange(trip, '(B L) N E -> B L N E', B=B)
                trip = trip.mean(2)
            else:
                raise NotImplementedError(self.mt_strategy)

        pos_x = torch.stack([embed(pos[..., i]) for i, embed in enumerate(self.pos_embeds)], -1).sum(-1)
        out = self.transformer(trip + pos_x, mask=src_mask.bool(), src_key_padding_mask=batch_mask.bool())
        out = self.out_linear(out)

        if multi_type:
            if self.mt_strategy == 'flat':
                out = rearrange(out, 'B (L N) E -> B L N E', N=N_type)
            elif self.mt_strategy in ['fc', 'mean', 'attention']:
                if self.repeat_output:
                    out = repeat(out, 'B L E -> B L N E', N=N_type)
                else:
                    out = rearrange(out, 'B L (N E) -> B L N E', N=N_type)

        return out
