import math
import os
import shutil
from os.path import exists
from datetime import datetime
import torch

import numpy as np


def create_if_noexists(*paths):
    """ Create directories if they don't exist already. """
    for path in paths:
        if not exists(path):
            os.makedirs(path)


def remove_if_exists(path):
    """ Remove a file if it exists. """
    if exists(path):
        os.remove(path)


def gather_all_param(*models):
    """ Gather all learnable parameters in the models as a list. """
    parameters = []
    for encoder in models:
        parameters += list(encoder.parameters())
    return parameters


def next_batch(data, batch_size):
    """ Yield the next batch of given data. """
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        if end_index - start_index > 1:
            yield data[start_index:end_index]


def clean_dirs(*dirs):
    """ Remove the given directories, including all contained files and sub-directories. """
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def intersection(lst1, lst2):
    """ Calculates the intersection of two sets, or lists. """
    lst3 = list(set(lst1) & set(lst2))
    return lst3


def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 't' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def cal_model_size(model):
    """ Calculate the total size (in megabytes) of a torch module. """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def geo_distance(lng1, lat1, lng2, lat2):
    """ Calculcate the geographical distance between two points (or one target point and an array of points). """
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
    return distance


def spherical_distance(coor_array):
    """
    Calculate spherical distance given the coordinate array.

    :param coor_array: the coordiante array with shape (N, 4), each row are (lng1, lat1, lng2, lat2).
    """
    lng1, lat1, lng2, lat2 = torch.chunk(coor_array, 4, -1)
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    distance = 2*torch.asin(torch.sqrt(a))*6371*1000
    distance = torch.round(distance/1000, decimals=3)
    return distance.squeeze(-1)