import sys

sys.path.append('../')

import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
import lmdb
CodeRow1 = namedtuple('CodeRow1', ['id', 'filename'])


class LMDB_VQVAE_PIXEL_Dataset(Dataset):
    def __init__(self, path, x_name, cond_name):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.x_name = x_name
        self.cond_name = cond_name
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))
        if self.cond_name is None:
            return torch.from_numpy(row[self.x_name]), None, row.filename
        return torch.from_numpy(row[self.x_name]), torch.from_numpy(row[self.cond_name]), row.filename
