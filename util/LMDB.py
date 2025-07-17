import os
import lmdb
import torch
import pickle
from typing import Union

class LMDBEEGSignalIO():

    def __init__(self, io_path: str, io_size: int = 1048576) -> None:
        self.io_path = io_path
        self.io_size = io_size
        self.write_pointer = 0
        if not os.path.exists(self.io_path):
            os.makedirs(self.io_path, exist_ok=True)
            self.database = lmdb.open(path=self.io_path, map_size=self.io_size, lock=False)
        else:
            self.database = lmdb.open(path=self.io_path, readonly=True, lock=False)

    def __del__(self):
        if hasattr(self, 'database'):
            self.database.close()

    def __len__(self):
        with self.database.begin(write=False) as transaction:
            return transaction.stat()['entries']

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:

        if key is None:
            key = str(self.write_pointer)
            self.write_pointer += 1

        if eeg is None:
            raise RuntimeError(f'EEG data is null!')

        try_again = False
        try:
            with self.database.begin(write=True) as transaction:
                transaction.put(key.encode(), pickle.dumps(eeg))
        except lmdb.MapFullError:
            self.io_size = self.io_size * 2
            self.database.set_mapsize(self.io_size)
            try_again = True
        if try_again:
            return self.write_eeg(key=key, eeg=eeg)
        return key

    def read_eeg(self, key: str) -> any:
        with self.database.begin(write=False) as transaction:
            eeg = transaction.get(key.encode())
        if eeg is None:
            raise RuntimeError(f'read failure {key}!')
        return pickle.loads(eeg)

    def keys(self):
        with self.database.begin(write=False) as transaction:
            return [key.decode() for key in transaction.cursor().iternext(keys=True, values=False)]

    def eegs(self):
        return [self.read_eeg(key) for key in self.keys()]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['database']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.database = lmdb.open(path=self.io_path, map_size=self.io_size, lock=False)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items() if k != 'database'
        })
        result.database = lmdb.open(path=self.io_path, map_size=self.io_size, lock=False)
        return result