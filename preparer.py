import numpy as np
import torch

from pretrain.losses import trajsim


class BatchPreparer:
    """
    Prepare functions for different types of batch data.
    """

    @staticmethod
    def fetch_prepare_func(meta_type):
        """ Fetch a specific prepare function based on the meta type. """
        if 'gtm-' in meta_type:
            return BatchPreparer.prepare_gtm_batch
        else:
            raise NotImplementedError('No prepare function for meta type "' + meta_type + '".')

    @staticmethod
    def prepare_gtm_batch(batch_meta, device):
        trip, src, tgt, trip_len, src_len = batch_meta
        trip, src, tgt = (torch.from_numpy(np.stack(a, 0)).float().to(device) for a in [trip, src, tgt])
        trip_len, src_len = [torch.tensor(a).long().to(device) for a in [trip_len, src_len]]

        return trip, src, tgt, trip_len, src_len
