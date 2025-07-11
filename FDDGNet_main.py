"""
程序主文件。
"""
import os

import torch

from Path import Path
from train.FDDGNetTrainer import FDDGNetTrainer
from util.dataslice.amigosDataSlice import AmigosDataSlice
from util.dataslice.dreamerDataSlice import DreamerDataSlice
from util.trainmethod.generateFormat import GenerateFormat
from dataloader.mixtureDatasetDataLoader import MixtureDatasetDataLoader


def main():
    """
    程序入口，定义程序运行流程。
    """
    input_size = 14
    hidden_size = 512
    seq_len = 128
    classNumber = 2
    # 生成对应的lmdb数据库文件和对应的信息表
    if not os.path.isdir(Path.AMIGOS_SLICED_DATA_DIR):
        AmigosDataSlice.load_data(Path.AMIGOS_P_EEG_PATH,seq_len)
    if not os.path.isdir(Path.DREAMER_SLICED_DATA_DIR):
        DreamerDataSlice.load_data(Path.DREAMER_P_EEG_PATH,seq_len)
    # 指定参与训练的数据集并生成对应的训练及测试表
    # amigos     dreamer    amigos&dreamer
    datasets = 'amigos'
    file_path = Path.FORMAT_DIR + datasets
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        os.makedirs(file_path + '/los')
        os.makedirs(file_path + '/random')
        os.makedirs(file_path + '/clos')
        GenerateFormat(datasets, file_path).generate_leave_one_subject_format()
        GenerateFormat(datasets, file_path).generate_random_divide_format()
        GenerateFormat(datasets, file_path).generate_cross_dataset_leave_one_subject_format()

    subject_num = 2
    skips = []

    if datasets == 'amigos':
        # 35
        subject_num = 35
        skips = []
    if datasets == 'dreamer':
        subject_num = 24
        skips = []
    if datasets == 'amigos&dreamer':
        # 2
        subject_num = 3
        skips = []


    for i in range(1,subject_num):
        if i not in skips:
            subject_id = i
            # 定义训练集及测试集数据加载器
            train_dataset = MixtureDatasetDataLoader(subject_id, 'train', 'los', file_path)
            test_dataset = MixtureDatasetDataLoader(subject_id, 'test', 'los', file_path)

            domainNumber = len(train_dataset.subject_dataset_mapping)

            # 调用模型训练器
            model_trainer = FDDGNetTrainer(input_size, hidden_size, seq_len, classNumber, domainNumber, datasets, subject_id)

            # 模型训练
            model_trainer.train(train_dataset, test_dataset)

            # 清理数据加载器和模型
            del train_dataset
            del test_dataset
            del model_trainer
            # 清理显存
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
