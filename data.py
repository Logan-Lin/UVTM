import os
from collections import Counter, defaultdict
import json
import csv

import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors
from einops import repeat

from utils import create_if_noexists, geo_distance


pd.options.mode.chained_assignment = None
CLASS_COL = 'driver'
SET_NAMES = [(0, 'train'), (1, 'val'), (2, 'test')]
MIN_TRIP_LEN = 6
MAX_TRIP_LEN = 120
TARGET_SAMPLE_RATE = 15
TRIP_COLS = ['tod', 'road', 'road_prop', 'lng', 'lat', 'weekday', 'seq_i']

with open('path_conf.json') as fp:
    conf = json.load(fp)


class Data:
    def __init__(self, name):
        self.name = name
        self.small = 'small' in name

        paths = conf['small'] if self.small else conf['full']
        self.base_path = paths['meta_path']
        self.dataset_path = paths['dataset_path']

        self.df_path = f'{self.dataset_path}/{self.name}.h5'
        self.meta_dir = f'{self.base_path}/meta/{self.name}'
        self.stat_path = f'{self.meta_dir}/stat.h5'

        self.get_meta_path = lambda meta_type, select_set: os.path.join(
            self.meta_dir, f'{meta_type}_{select_set}.npz')

    """ Load functions for loading dataframes and meta. """

    def read_hdf(self):
        # Load the raw data from HDF files.
        # One set of raw dataset is composed of one HDF file with four keys.
        # The trips contains the sequences of trajectories, with three columns: trip, time, road
        self.trips = pd.read_hdf(self.df_path, key='trips')
        # The trip_info contains meta information about trips. For now, this is mainly used for class labels.
        self.trip_info = pd.read_hdf(self.df_path, key='trip_info')
        # The road_info contains meta information about roads.
        self.road_info = pd.read_hdf(self.df_path, key='road_info')
        # self.trips = pd.merge(self.trips, self.road_info[['road', 'lng', 'lat']], on='road', how='left')

        # Add some columns to the trip
        self.trips['seconds'] = self.trips['time'].apply(lambda x: x.timestamp())
        self.trips['minutes'] = self.trips['time'].apply(lambda x: x.timestamp() / 60)
        self.trips['tod'] = self.trips['time'].apply(lambda x: x.timestamp() % (24 * 60 * 60) / (24 * 60 * 60))
        self.trips['weekday'] = self.trips['time'].dt.weekday
        self.stat = self.trips.describe()

        num_road = int(self.road_info['road'].max() + 1)
        num_class = int(self.trip_info[CLASS_COL].max() + 1)
        self.data_info = pd.Series([num_road, num_class], index=['num_road', 'num_class'])
        print('Loaded DataFrame from', self.df_path)

        num_trips = self.trip_info.shape[0]
        self.train_val_test_trips = (self.trip_info['trip'].iloc[:int(num_trips * 0.8)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.8):int(num_trips * 0.9)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.9):])

        create_if_noexists(self.meta_dir)
        self.stat.to_hdf(self.stat_path, key='stat')
        self.data_info.to_hdf(self.stat_path, key='info')
        print(self.stat)
        print(self.data_info)
        print('Dumped dataset info into', self.stat_path)

        self.valid_trips = [self.get_valid_trip_id(i) for i in trange(3, desc='Getting all valid trips')]
        print('Total number of valid trips:', sum([len(l) for l in self.valid_trips]))

    def load_stat(self):
        # Load statistical information for features.
        self.stat = pd.read_hdf(self.stat_path, key='stat')
        self.data_info = pd.read_hdf(self.stat_path, key='info')

    def load_meta(self, meta_type, select_set):
        meta_path = self.get_meta_path(meta_type, select_set)
        loaded = np.load(meta_path, allow_pickle=True)
        # print('Loaded meta from', meta_path)
        return list(loaded.values())

    def get_valid_trip_id(self, select_set):
        select_trip_id = self.train_val_test_trips[select_set]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        valid_trip_id = []
        for _, group in tqdm(trips.groupby('trip'), desc='Filtering trips', total=select_trip_id.shape[0], leave=False):
            if (not group.isna().any().any()) and group.shape[0] >= MIN_TRIP_LEN and group.shape[0] <= MAX_TRIP_LEN:
                if ((group['seconds'] - group.shift(1)['seconds']).iloc[1:] == TARGET_SAMPLE_RATE).all():
                    valid_trip_id.append(group.iloc[0]['trip'])
        return valid_trip_id

    def dump_meta(self, meta_type, select_set, ext_trip_name=None, ext_arr_name=None, ext_lng_first=False):
        """
        Dump meta data into numpy binary files for fast loading later.

        :param meta_type: type name of meta data to dump.
        :param select_set: index of set to dump. 0 - training set, 1 - validation set, and 2 - testing set.
        """
        # Prepare some common objects that will probably be useful for various types of meta data.
        select_trip_id = self.valid_trips[select_set]
        known_trips = self.trips[self.trips['trip'].isin(self.valid_trips[0] + self.valid_trips[1])]
        set_name = SET_NAMES[select_set][1]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        trip_info = self.trip_info[self.trip_info['trip'].isin(select_trip_id)]
        max_trip_len = max(Counter(trips['trip']).values())
        trip_normalizer = Normalizer(self.stat, feat_cols=[0, 2, 3, 4], norm_type='minmax')

        if 'gtm-' in meta_type:
            params = meta_type.split('-')
            sample_rate = float(params[1])
            shuffle_target = bool(int(params[2][1:]))
            if len(params) > 3:
                special_type = params[3]
            else:
                special_type = 'none'

            trip_arr, source_arr, target_arr, trip_len, source_len = ([] for _ in range(5))
            # A set of <Known, Start, End, Mask> tokens, for spatial, temporal and roadnet.
            token_start = self.data_info['num_road'].item()
            known_col = [token_start, token_start, token_start]
            start_col = [token_start+1, token_start+1, token_start+1] + [-999] * 6
            end_col = [token_start+2, token_start+2, token_start+2] + [-999] * 4
            mask_col = [token_start+3, token_start+3, token_start+3] + [-999] * 5

            for _, group in tqdm(trips.groupby('trip'), desc=f'Gathering GLM sequences, type: {special_type}',
                                 total=len(select_trip_id), leave=False):
                retrip = self.resample_trip(group, sample_rate)

                group = group[TRIP_COLS + ['minutes']].to_numpy()
                group = trip_normalizer(group)  # (L_trip, N_feat)

                trip_row, source_row, target_row = ([] for _ in range(3))
                trip_i = 1
                if special_type == 'tte':
                    resampled_seq_i = [retrip['seq_i'].iloc[0], retrip['seq_i'].iloc[-1]] + [group.shape[0]]
                # elif special_type == 'tp':
                #     resampled_seq_i = retrip['seq_i'].tolist()[:-1] + [group.shape[0]]
                else:
                    resampled_seq_i = retrip['seq_i'].tolist() + [group.shape[0]]

                def _get_trip_src_tgt(trip_span, trip_i, i, mask_trip):
                    span_len = trip_span.shape[0]
                    if mask_trip:
                        trip_item = mask_col + [trip_i, 0]
                    else:
                        # Three tokens (spatial, temporal, roadnet), weekday, tod, lng, lat, seq_i, two positions
                        trip_item = known_col + trip_span[0][[5, 0, 3, 4, 6]].tolist() + [trip_i, 0]
                    # Three tokens (spatial, temporal, roadnet), weekday, tod, lng, lat, road_prop, seq_i, two positions
                    src_span = np.concatenate([np.ones((span_len, 2)) * token_start,
                                               trip_span[:, [1, 5, 0, 3, 4, 2, 6]],
                                               np.array([[trip_i] * span_len, list(range(2, span_len + 2))]).transpose(1, 0)], -1)
                    # Three tokens (spatial, temporal, roadnet), lng, lat, road_prop, offset (minutes)
                    tgt_span = np.concatenate([np.ones((span_len, 2)) * -999,
                                               trip_span[:, [1, 3, 4, 2]],
                                               trip_span[:, [7]] - group[0, 7]], -1)

                    if special_type == 'tte' and trip_i != 1:
                        trip_item[1] = token_start+3
                        trip_item[4] = -999
                    elif special_type == 'tp' and i == len(resampled_seq_i) - 2:
                        trip_item[0] = token_start + 3
                        trip_item[5] = -999
                        trip_item[6] = -999

                    src = np.concatenate([np.array([start_col + [trip_i, 1]]), src_span], 0)
                    tgt = np.concatenate([tgt_span, np.array([end_col])], 0)
                    return trip_item, src, tgt

                for i, (l, r) in enumerate(zip(resampled_seq_i[:-1], resampled_seq_i[1:])):
                    trip_span = group[l:r]
                    trip, src, tgt = _get_trip_src_tgt(trip_span[:1], trip_i, i, False)
                    trip_row.append(trip)
                    source_row.append(src)
                    target_row.append(tgt)
                    trip_i += 1

                    if trip_span.shape[0] > 1:
                        trip, src, tgt = _get_trip_src_tgt(trip_span[1:], trip_i, i, True)
                        trip_row.append(trip)
                        if special_type == 'tte':
                            pass
                        else:
                            source_row.append(src)
                            target_row.append(tgt)
                        trip_i += 1

                if shuffle_target:
                    source_row, target_row = shuffle(source_row, target_row)
                trip_row = np.array(trip_row)
                source_row, target_row = (np.concatenate(row, 0) for row in [source_row, target_row])

                trip_len.append(trip_row.shape[0]), source_len.append(source_row.shape[0])
                trip_arr.append(trip_row), source_arr.append(source_row), target_arr.append(target_row)

            trip_len, source_len = np.array(trip_len), np.array(source_len)
            trip_arr, source_arr, target_arr = (np.stack([np.concatenate([row, np.ones((max_len - row.shape[0], row.shape[1])) * -999], 0)
                                                          for row in arr])
                                                for arr, max_len in zip([trip_arr, source_arr, target_arr],
                                                                        [trip_len.max(), source_len.max(), source_len.max()]))
            meta = [trip_arr, source_arr, target_arr, trip_len, source_len]

        elif 'distcand' in meta_type:
            """
            The "distcand" meta data records the distance candidates given a certain distance threshold in meters.
            The road segments within the distance threshold of GPS points are recorded.
            """
            # Parameters are given in the meta_type. Use format "distcand-{dist_thres}".
            dist_thres = float(meta_type.split('-')[1])

            # Prepare the coordinates of roads.
            road_coors = self.road_info[['road', 'road_lng', 'road_lat']].to_numpy()
            cand_seq = []
            max_num_cand = 0
            for _, group in tqdm(trips.groupby('trip'), desc='Finding distance candidates', total=len(select_trip_id), leave=False):
                cand_row = []
                for _, row in group.iterrows():
                    # Calculate the geographical distance between this GPS point and all road segments.
                    dist = geo_distance(row['lng'], row['lat'], road_coors[:, 1], road_coors[:, 2])
                    cand = road_coors[:, 0][dist <= dist_thres].astype(int).tolist()
                    max_num_cand = max(len(cand), max_num_cand)
                    cand_row.append(cand)
                cand_seq.append(cand_row)

            pad_seq = []
            for cand_row in cand_seq:
                cand_row = [cand + [-999] * (max_num_cand - len(cand)) for cand in cand_row]
                pad_seq.append(cand_row + [[-999] * max_num_cand] * (max_trip_len - len(cand_row)))
            pad_seq = np.array(pad_seq).astype(int)

            meta = [pad_seq]

        elif 'knncand' in meta_type:
            """
            The "knncand" meta data records the k-nearest neighbor candidates given a certain k value.
            The road segments that are within the set of kNN of GPS points are recorded.
            """
            # Parameters are given in the meta_type. Use format "knncand-{k}".
            k = int(meta_type.split('-')[1])

            # Prepare the coordinates of roads and the knn model.
            road_coors = self.road_info[['road', 'road_lng', 'road_lat']].to_numpy()
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(road_coors[:, [1, 2]])
            cand_seq = []
            for _, group in tqdm(trips.groupby('trip'), desc='Finding knn candidates', total=len(select_trip_id), leave=False):
                _, neighbors = knn.kneighbors(group[['lng', 'lat']].to_numpy())
                cand_seq.append(neighbors)

            pad_seq = []
            for cand_row in cand_seq:
                pad_seq.append(np.concatenate([cand_row, np.repeat(
                    cand_row[-1:], max_trip_len - cand_row.shape[0], 0)], 0))
            pad_seq = np.stack(pad_seq, 0).astype(int)

            meta = [pad_seq]

        else:
            raise NotImplementedError('No meta type', meta_type)

        create_if_noexists(self.meta_dir)
        if ext_trip_name is not None:
            meta_type += f'_{ext_trip_name}'
        if ext_arr_name is not None:
            meta_type += f'_{ext_arr_name}'
        meta_path = self.get_meta_path(meta_type, select_set)
        np.savez(meta_path, *meta)

    @staticmethod
    def resample_trip(trip, sample_rate):
        if sample_rate > 0:
            resampled_trip = trip.reset_index().set_index('time').resample(
                rule=pd.Timedelta(sample_rate, 'seconds'), origin='start').asfreq()
            resampled_trip = resampled_trip.reset_index()
            if resampled_trip['seq_i'].iloc[-1] != trip['seq_i'].iloc[-1]:
                resampled_trip = pd.concat([resampled_trip, trip.iloc[[-1]]], axis=0)
        else:
            resampled_trip = trip
        return resampled_trip


class Normalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None, norm_type='zscore'):
        super().__init__()

        self.stat = stat
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]
        self.norm_type = norm_type

    def _norm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = (x_col - self.stat.loc['mean', col_name]) / self.stat.loc['std', col_name]
        elif self.norm_type == 'minmax':
            x_col = (x_col - self.stat.loc['min', col_name]) / \
                (self.stat.loc['max', col_name] - self.stat.loc['min', col_name])
            x_col = x_col * 2 - 1
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    def forward(self, arr):
        """ Normalize the input array. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            x[..., col] = self._norm_col(x[..., col], name)
        return x


class Denormalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None, norm_type='zscore'):
        super().__init__()

        self.stat = stat
        self.feat_names = feat_names
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]
        self.norm_type = norm_type

    def _denorm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = x_col * self.stat.loc['std', col_name] + self.stat.loc['mean', col_name]
        elif self.norm_type == 'minmax':
            x_col = (x_col + 1) / 2
            x_col = x_col * (self.stat.loc['max', col_name] - self.stat.loc['min', col_name]) + \
                self.stat.loc['min', col_name]
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    def forward(self, select_cols, arr):
        """ Denormalize the input batch. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            if select_cols is None:
                x = self._denorm_col(x, name)
            else:
                if col in select_cols:
                    x[..., col] = self._denorm_col(x[..., col], name)
        return x


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-n', '--name', help='the name of the dataset', type=str, required=True)
    parser.add_argument('-t', '--types', help='the type of meta data to dump', type=str, required=True)
    parser.add_argument('-i', '--indices', help='the set index to dump meta', type=str)
    parser.add_argument('--trip', type=str, help='name of external trip HDF', default=None)
    parser.add_argument('--arr', type=str, help='name of external arr npy', default=None)
    parser.add_argument('--lngfirst', action='store_true',
                        help='whether the lng column is the first column in external arr npy')

    args = parser.parse_args()

    for data_name in args.name.split(','):
        data = Data(data_name)
        data.read_hdf()
        with tqdm(args.types.split(',')) as bar:
            for type in bar:
                bar.set_description(f'Dumping meta {type}')
                for i in args.indices.split(','):
                    data.dump_meta(type, int(i), args.trip, args.arr, args.lngfirst)
