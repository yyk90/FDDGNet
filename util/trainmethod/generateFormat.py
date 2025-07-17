"""
留一被试法。
"""
import csv
from Path import Path


class GenerateFormat:
    """
    为留一被试法生成对应的测试集、训练集信息表。
    """

    def __init__(self, datasets, los_file_path):
        """
        datasets: 参与留一被试实验的数据集。不同数据集请使用 & 连接，形如：dreamer & amigos。
        """
        datasets = [item.strip() for item in datasets.split('&')]
        valid_datasets = ['dreamer', 'amigos', 'gameemo','seed','amigos2_DE', 'dreamer_DE','selfdata']
        # 检验输入的数据集名称是否正确
        for dataset in datasets:
            if dataset not in valid_datasets:
                raise ValueError(f"无效的数据集: {dataset}")
        # 数据库列表
        self.datasets = datasets
        # 存储全部数据集信息的字典（键：新编被试编号/值：row列表）
        self.datasets_msg_dict = {}
        # 创建留一被试表路径
        self.los_file_path = los_file_path

    def create_temporary_dict(self):
        """
        根据用户指定的数据集组建字典。
        """
        index = 1
        for dataset in self.datasets:
            if dataset == 'dreamer' or dataset == 'dreamer_DE':
                for i in range(1, 24):
                    subject_msg_list = []
                    file_path = Path.DREAMER_SLICED_DATA_DIR + '_DE/Subject_' + str(i) + '/msg.csv'
                    # 读取被试的信息表格并存储到subject_msg_list中
                    with open(file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        # 跳过表头
                        next(reader)
                        # 写入信息
                        for row in reader:
                            subject_msg_list.append(row)
                        self.datasets_msg_dict[index] = subject_msg_list
                        index += 1
            if dataset == 'amigos' or dataset == 'amigos2_DE':
                for i in range(1, 41):
                    subject_msg_list = []
                    # 跳过有问题的被试
                    # torcheeg 中给出有问题的试次编号
                    # if i in [9,12,21,22,23,24,32,33]:
                    # 自己处理的有问题的试次编号
                    if i in [11, 16, 23, 24, 32, 33]:
                        continue
                    file_path = Path.AMIGOS_SLICED_DATA_DIR + '_DE/Subject_' + str(i) + '/msg.csv'
                    # 读取被试的信息表格并存储到subject_msg_list中
                    with open(file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        # 跳过表头
                        next(reader)
                        # 写入信息
                        for row in reader:
                            subject_msg_list.append(row)
                        self.datasets_msg_dict[index] = subject_msg_list
                        index += 1
            if dataset == 'gameemo':
                for i in range(1, 29):
                    subject_msg_list = []
                    file_path = Path.GAMEEMO_SLICED_DATA_DIR + '/Subject_' + str(i) + '/msg.csv'
                    # 读取被试的信息表格并存储到subject_msg_list中
                    with open(file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        # 跳过表头
                        next(reader)
                        # 写入信息
                        for row in reader:
                            subject_msg_list.append(row)
                        self.datasets_msg_dict[index] = subject_msg_list
                        index += 1
            if dataset == 'seed':
                for i in range(1, 16):
                    subject_msg_list = []
                    for j in range(1,4):

                        file_path = Path.SEED_SLICED_DATA_DIR + '/Subject_' + str(i) + '_' + str(j) + '/msg.csv'
                        # 读取被试的信息表格并存储到subject_msg_list中
                        with open(file_path, 'r') as csvfile:
                            reader = csv.reader(csvfile)
                            # 跳过表头
                            next(reader)
                            # 写入信息
                            for row in reader:
                                subject_msg_list.append(row)
                    self.datasets_msg_dict[index] = subject_msg_list
                    index += 1

            if dataset == 'selfdata':
                for i in range(1, 2):
                    subject_msg_list = []
                    # 跳过有问题的被试
                    # torcheeg 中给出有问题的试次编号
                    # if i in [9,12,21,22,23,24,32,33]:
                    # 自己处理的有问题的试次编号
                    file_path = Path.SELFDATA_SLICED_DATA_DIR + '/Subject_' + str(i) + '/msg.csv'
                    # 读取被试的信息表格并存储到subject_msg_list中
                    with open(file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        # 跳过表头
                        next(reader)
                        # 写入信息
                        for row in reader:
                            subject_msg_list.append(row)
                        self.datasets_msg_dict[index] = subject_msg_list
                        index += 1

    def generate_leave_one_subject_format(self):
        """
        生成留一被试训练及测试表。
        """
        # 生成数据集字典
        self.create_temporary_dict()
        # 创建模型训练需要的表格
        for i in self.datasets_msg_dict.keys():
            for key, value in self.datasets_msg_dict.items():
                if key == i:
                    # 生成测试集信息表
                    with open(self.los_file_path + '/los/test_' + str(i) + '.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        # 插入表头
                        if file.tell() == 0:
                            if 'seed' in self.datasets:
                                writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'scession_id','trial_id', 'label', 'dataset'])
                            else:
                                writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                                 'valence', 'arousal', 'label', 'dataset'])
                        for row in value:
                            writer.writerow(row)
                else:
                    # 生成训练集信息表
                    with open(self.los_file_path + '/los/train_' + str(i) + '.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        # 插入表头
                        if file.tell() == 0:
                            if 'seed' in self.datasets:
                                writer.writerow(
                                    ['start_at', 'end_at', 'clip_id', 'subject_id', 'scession_id', 'trial_id', 'label',
                                     'dataset'])
                            else:
                                writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                                 'valence', 'arousal', 'label', 'dataset'])
                        for row in value:
                            writer.writerow(row)

    def generate_random_divide_format(self):
        """
        生成随机划分法训练及测试表。（在验证实验中需要使用到随机划分法）
        """
        # 生成数据集字典
        self.create_temporary_dict()
        # 创建模型训练需要的表格
        for key, value in self.datasets_msg_dict.items():
            # 分界点
            split_point = int(len(value) * 0.8)
            # 生成测试集信息表
            with open(self.los_file_path + '/random/train.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                # 插入表头
                if file.tell() == 0:
                    if 'seed' in self.datasets:
                        writer.writerow(
                            ['start_at', 'end_at', 'clip_id', 'subject_id', 'scession_id', 'trial_id', 'label',
                             'dataset'])
                    else:
                        writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                         'valence', 'arousal', 'label', 'dataset'])
                for i in range(split_point):
                    writer.writerow(value[i])
            # 生成训练集信息表
            with open(self.los_file_path + '/random/test.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                # 插入表头
                if file.tell() == 0:
                    if 'seed' in self.datasets:
                        writer.writerow(
                            ['start_at', 'end_at', 'clip_id', 'subject_id', 'scession_id', 'trial_id', 'label',
                             'dataset'])
                    else:
                        writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                         'valence', 'arousal', 'label', 'dataset'])
                for i in range(split_point, len(value)):
                    writer.writerow(value[i])

    def generate_cross_dataset_leave_one_subject_format(self):
        """
        生成跨数据集留一被试训练及测试表。
        """
        # 生成数据集字典
        self.create_temporary_dict()
        # 将指定的数据集拼合作为训练集
        for key, value in self.datasets_msg_dict.items():
            with open(self.los_file_path + '/clos/train.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                # 插入表头
                if file.tell() == 0:
                    if 'seed' in self.datasets:
                        writer.writerow(
                            ['start_at', 'end_at', 'clip_id', 'subject_id', 'scession_id', 'trial_id', 'label',
                             'dataset'])
                    else:
                        writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                         'valence', 'arousal', 'label', 'dataset'])
                for row in value:
                    writer.writerow(row)
        # 重置信息，设置gameemo作为测试集
        self.datasets = ['gameemo']
        self.datasets_msg_dict = {}
        # 生成测试集字典
        self.create_temporary_dict()
        for key, value in self.datasets_msg_dict.items():
            with open(self.los_file_path + '/clos/test.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                # 插入表头
                if file.tell() == 0:
                    if 'seed' in self.datasets:
                        writer.writerow(
                            ['start_at', 'end_at', 'clip_id', 'subject_id', 'scession_id', 'trial_id', 'label',
                             'dataset'])
                    else:
                        writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                         'valence', 'arousal', 'label', 'dataset'])
                for row in value:
                    writer.writerow(row)
