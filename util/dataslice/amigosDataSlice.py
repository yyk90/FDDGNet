"""
模型训练所需的数据加载器文件。定义amigos数据集的数据加载器。
"""
import re
import os
import csv
import xlrd
import scipy.io as scio
from Path import Path
from util.dataslice.LMDB import LMDBEEGSignalIO


class AmigosDataSlice:
    """
    amigos数据集的dataloader。
    """

    @staticmethod
    def load_data(file_path,seq_len):
        """
        导入目标文件夹的数据并进行处理。
        此方法会生成lmdb数据库文件以及信息表，用以模型训练。
        """
        # 记录各类样本数
        positive = 0
        negative = 0
        invalid = 0
        # 创建根目录
        os.makedirs(Path.AMIGOS_SLICED_DATA_DIR)
        # 加载标签文件
        label = AmigosDataSlice.load_labels()
        # 加载P-EEG数据
        all_mat_file = os.walk(file_path)
        skip_set = {'AmigosLabel.xls'}
        #skip_subject = {'9', '12', '21', '22', '23', '24', '33'}
        #skip_subject = {'16', '23', '24', '32', '33'}
        skip_subject = { }
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                if file_name not in skip_set:
                    # 获取被试编号
                    subject_id = ''.join(re.findall(r'\d+', file_name.split("_")[1]))
                    # 去除掉数据质量不好的被试
                    if subject_id not in skip_subject:
                        # 创建文件路径
                        dir_path = Path.AMIGOS_SLICED_DATA_DIR + '/Subject_' + subject_id
                        # 开启数据库
                        database = LMDBEEGSignalIO(dir_path)
                        # 编辑csv表头
                        header = ['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                  'valence', 'arousal', 'label', 'dataset']
                        with open(dir_path+'/msg.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(header)
                        # 加载被试数据
                        subject_data = scio.loadmat(file_path + "/" + file_name)
                        #print(label)
                        for i in range(16):
                            # 获取当前试次EEG数据
                            subject_eeg_trail = subject_data['time' + str(i + 1)]
                            # 获取当前试次的标签
                            arousal = label[(int(subject_id) - 1) * 16 + i][1]
                            valence = label[(int(subject_id) - 1) * 16 + i][2]
                            quality = label[(int(subject_id) - 1) * 16 + i][3]
                            valence_temp = float(valence)
                            quality = int(quality)
                            # 赋予离散标签
                            if quality == 0:
                                if valence_temp >= 5.0:
                                    # 正向情绪
                                    discrete_label = 1
                                    positive = positive + 1
                                elif valence_temp < 5.0:
                                    # 负面情绪
                                    discrete_label = 0
                                    negative = negative + 1
                                else:
                                    # 界限不明显的数据标为无效
                                    discrete_label = 999
                                    invalid = invalid + 1
                            else:
                                # 数据质量差的数据标为无效
                                discrete_label = 999
                            # 对数据进行切片
                            start_at = 0
                            while start_at + seq_len < subject_eeg_trail.shape[1]:
                                # 将数据以秒为单位切片并存储
                                end_at = start_at + seq_len
                                second_eeg = subject_eeg_trail[:, start_at:end_at]
                                # 将切片数据存入数据库并返回clip_id
                                clip_id = database.write_eeg(second_eeg)
                                # 将切片数据信息写入表格
                                data = [start_at, end_at, clip_id, subject_id, i, valence, arousal, discrete_label, 1]
                                with open(dir_path + '/msg.csv', 'a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow(data)
                                start_at = end_at
                            print("amigos数据切片中...请等待...")
        print("amigos各类样本量（正、负、无效）：")
        print(positive)
        print(negative)
        print(invalid)

    @staticmethod
    def load_labels():
        """
        加载标签文件。
        :return: 标签数组。
        """
        label = []
        file_path = Path.AMIGOS_LABELS
        workbook = xlrd.open_workbook(file_path)
        sheet = workbook.sheet_by_index(0)
        # 从第三行开始遍历每一行
        for row_idx in range(2, sheet.nrows):
            row = sheet.row_values(row_idx)
            # 检查行是否为空
            if not all(cell == '' for cell in row):
                label.append(row)
        return label

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
