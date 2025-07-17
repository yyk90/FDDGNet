"""
数据加载器。
"""
import csv
import numpy as np

from Path import Path
from torch.utils.data import Dataset
from util.dataslice.LMDB import LMDBEEGSignalIO


class MixtureDatasetDataLoader(Dataset):
    """
    混合数据集的数据加载器。
    """

    def __init__(self, current_subject, train_or_test, random_or_los_or_clos, file_path):
        """
        :param current_subject: 目标域被试编号。
        :param train_or_test: 训练集/测试集。
        :param random_or_los_or_clos: 随机划分数据集/留一被试法。
        :param file_path: 文件路径。
        """
        self.info_list = None
        self.current_subject = current_subject
        self.subject_dataset_mapping = {}
        # 加载留一信息表
        if train_or_test == 'train':
            if random_or_los_or_clos == 'los':
                leave_one_subject_file = file_path + '/los/train_' + str(self.current_subject) + '.csv'
            elif random_or_los_or_clos == 'random':
                leave_one_subject_file = file_path + '/random/train.csv'
            elif random_or_los_or_clos == 'clos':
                leave_one_subject_file = file_path + '/clos/train.csv'
            else:
                raise ValueError(f"无效的指令: {random_or_los_or_clos}")
        elif train_or_test == 'test':
            if random_or_los_or_clos == 'los':
                leave_one_subject_file = file_path + '/los/test_' + str(self.current_subject) + '.csv'
            elif random_or_los_or_clos == 'random':
                leave_one_subject_file = file_path + '/random/test.csv'
            elif random_or_los_or_clos == 'clos':
                leave_one_subject_file = file_path + '/clos/test.csv'
            else:
                raise ValueError(f"无效的指令: {random_or_los_or_clos}")
        else:
            raise ValueError(f"无效的指令: {train_or_test}")
        self.info_list = self.read_csv_file(leave_one_subject_file)
        # 过滤数据，剔除无效数据
        self.info_list = [item for item in self.read_csv_file(leave_one_subject_file) if int(item[7]) != 999]

        # 训练map
        # 创建一个字典，用于映射 subject_id 和 dataset_id 的组合到新的编号

        new_id = 0

        # 遍历 self.info_list，对 subject_id 进行重新编号
        for row in self.info_list:
            if len(self.info_list[0]) == 9:
                subject_id = row[3]  # 第四列是 subject_id
                dataset_id = row[8]  # 第九列是 dataset_id

            # 创建 subject_id 和 dataset_id 的组合键
            key = (subject_id, dataset_id)

            # 如果该组合还未出现过，给它分配一个新的编号
            if key not in self.subject_dataset_mapping:
                self.subject_dataset_mapping[key] = new_id
                new_id += 1

    @staticmethod
    def read_csv_file(path):
        """
        读取csv文件。
        :param path: 文件路径。
        :return: 数据列表。
        """
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过第一行表头
            result = list(reader)
        return result

    def __getitem__(self, index):
        """
        继承自Dataset的方法，实现后用以遍历数据集的数据。
        :param index: 编号，此程序中为留一被试表的行号。
        :return: eeg数据，离散标签，被试编号，数据集编号。
        """
        if len(self.info_list[0]) == 9:
            # 根据index查询信息表
            info = self.info_list[index]
            clip_id = info[2]
            subject_id = info[3]
            dataset_id = int(info[8])
            # 根据dataset_id和clip_id查找脑电数据文件
            dataset_path = None
            if dataset_id == 0:
                dataset_path = Path.DREAMER_SLICED_DATA_DIR
            elif dataset_id == 1:
                dataset_path = Path.AMIGOS_SLICED_DATA_DIR
            database_file = dataset_path + '/Subject_' + subject_id
            # 读取eeg
            eeg = LMDBEEGSignalIO(database_file).read_eeg(clip_id)
            eeg = np.array(eeg)
            eeg = np.transpose(eeg)
            # 读取标签
            label = int(info[7])
            trial_id = int(info[4])
            valence = float(info[5])
            arousal = float(info[6])

            key = (subject_id, str(dataset_id))
            subject_id_new = self.subject_dataset_mapping.get(key)

            return eeg, label, subject_id_new, trial_id, valence, arousal

    def __len__(self):
        """
        继承自Dataset的方法，返回数据集长度。
        :return: 数据集总长度，此程序中为留一被试表的长度。
        """
        return len(self.info_list) - 1
