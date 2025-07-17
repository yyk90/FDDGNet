"""
数据加载器所依赖的LMDB数据库逻辑，包括初始化、写入、读取等操作。
"""
import os
import lmdb
import torch
import pickle
from typing import Union


class LMDBEEGSignalIO():
    """
    LMDB交互类。用于操纵LMDB数据库。
    LMDB：全称Lightning Memory-Mapped Database，是一种高效的键值存储数据库，用于在内存映射的环境下提供高性能和低延迟的数据访问。
    """

    def __init__(self, io_path: str, io_size: int = 1048576) -> None:
        self.io_path = io_path
        self.io_size = io_size
        self.write_pointer = 0
        if not os.path.exists(self.io_path):
            os.makedirs(self.io_path, exist_ok=True)
            # 在指定路径下创建数据库
            self.database = lmdb.open(path=self.io_path, map_size=self.io_size, lock=False)
        else:
            # 如果文件已存在，则直接打开现有数据库
            self.database = lmdb.open(path=self.io_path, readonly=True, lock=False)

    def __del__(self):
        """
        析构函数，自动关闭LMDB数据库连接。
        """
        if hasattr(self, 'database'):
            self.database.close()

    def __len__(self):
        """
        返回LMDB数据库中存储的EEG信号数量。
        :return: LMDB数据库中存储的EEG信号数量。
        """
        with self.database.begin(write=False) as transaction:
            return transaction.stat()['entries']

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        """
        将EEG信号写入数据库。
        :param eeg: 准备写入数据库的 EEG 信号样本。
        :param key: 要插入的 EEG 信号的键值，如果未指定，则将自动生成一个递增的整数。
        :return: 数据库中已写入的 EEG 信号的索引。
        """
        if key is None:
            key = str(self.write_pointer)
            self.write_pointer += 1

        if eeg is None:
            raise RuntimeError(f'脑电数据为空!')

        try_again = False
        try:
            # 创建写事务，将序列化后的EEG数据写入数据库。
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
        """
        根据给定的键值从数据库中读取相应的EEG信号对象。
        :param key: 查询的键值。
        :return: eeg数据。
        """
        # 创建只读事务，从数据库中读取EEG数据。
        with self.database.begin(write=False) as transaction:
            eeg = transaction.get(key.encode())
        if eeg is None:
            raise RuntimeError(f'读取失败{key}!')
        # 反序列化EEG数据。
        return pickle.loads(eeg)

    def keys(self):
        """
        获取LMDB数据库中的所有键值，并以列表形式返回。
        :return: Key列表。
        """
        with self.database.begin(write=False) as transaction:
            return [key.decode() for key in transaction.cursor().iternext(keys=True, values=False)]

    def eegs(self):
        """
        获取LMDB数据库中的所有EEG信号对象，并以列表形式返回。
        :return: EEG列表。
        """
        return [self.read_eeg(key) for key in self.keys()]

    def __getstate__(self):
        """
        用于序列化对象，以便在使用pickle进行并行处理时能正确地保存对象状态。
        注：self.__dict__是对象的属性字典，用于存储对象的属性和对应的值。
        :return: 修改后的字典。
        """
        state = self.__dict__.copy()
        del state['database']
        return state

    def __setstate__(self, state):
        """
        用于反序列化对象，以便在使用pickle进行并行处理时能正确地恢复对象状态。
        :param state: 序列化的state。
        """
        self.__dict__.update(state)
        self.database = lmdb.open(path=self.io_path, map_size=self.io_size, lock=False)

    def __copy__(self):
        """
        返回当前对象的浅拷贝。
        :return: 具有相同属性值的新对象副本。
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items() if k != 'database'
        })
        result.database = lmdb.open(path=self.io_path, map_size=self.io_size, lock=False)
        return result