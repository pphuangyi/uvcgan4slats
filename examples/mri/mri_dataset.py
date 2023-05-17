import os

import numpy as np
from torch.utils.data import Dataset

from uvcgan.consts import SPLIT_TRAIN


def find_ndarrays_in_dir(path):
    result = []

    for fname in os.listdir(path):
        fullpath = os.path.join(path, fname)

        if not os.path.isfile(fullpath):
            continue

        ext = os.path.splitext(fname)[1]
        if ext != '.npz':
            continue

        result.append(fullpath)

    result.sort()
    return result


def load_ndarray(path):
    with np.load(path) as fh:
        return np.float32(fh[fh.files[0]])

# The class name must be Dataset
class Dataset(Dataset):

    def __init__(self, path, domain, split, **kwargs):

        super().__init__(**kwargs)

        self._path   = os.path.join(path, split, domain)
        self._fnames = find_ndarrays_in_dir(self._path)
        # normalization parameters

    def __len__(self):
        return len(self._fnames)

    def __getitem__(self, index):
        array = load_ndarray(self._fnames[index])
        array = 2 * array / 255. - 1.

        return np.expand_dims(array, 0)
