import os
import torch
from Path import Path
from train.FDDGNetTrainer import FDDGNetTrainer
from util.amigosDataSlice import AmigosDataSlice
from util.dreamerDataSlice import DreamerDataSlice
from util.generateFormat import GenerateFormat
from dataloader.mixtureDatasetDataLoader import MixtureDatasetDataLoader

def main():
    input_size = 14
    hidden_size = 512
    seq_len = 128
    classNumber = 2

    if not os.path.isdir(Path.AMIGOS_SLICED_DATA_DIR):
        AmigosDataSlice.load_data(Path.AMIGOS_P_EEG_PATH,seq_len)
    if not os.path.isdir(Path.DREAMER_SLICED_DATA_DIR):
        DreamerDataSlice.load_data(Path.DREAMER_P_EEG_PATH,seq_len)

    # amigos   or  dreamer  or  amigos&dreamer (can get with amigos{test1、train1} and dreamer{test1、train1})
    datasets = 'amigos'
    file_path = Path.FORMAT_DIR + datasets
    if not os.path.exists(file_path):
        os.makedirs(file_path)

        os.makedirs(file_path + '/los')

        GenerateFormat(datasets, file_path).generate_leave_one_subject_format()

    subject_num = 0
    skips = []

    if datasets == 'amigos':
        subject_num = 35
        skips = []
    if datasets == 'dreamer':
        subject_num = 24
        skips = []
    if datasets == 'amigos&dreamer':
        subject_num = 3
        skips = []

    for i in range(1,subject_num):
        if i not in skips:
            subject_id = i

            train_dataset = MixtureDatasetDataLoader(subject_id, 'train', 'los', file_path)
            test_dataset = MixtureDatasetDataLoader(subject_id, 'test', 'los', file_path)

            domainNumber = len(train_dataset.subject_dataset_mapping)

            model_trainer = FDDGNetTrainer(input_size, hidden_size, seq_len, classNumber, domainNumber, datasets, subject_id)

            model_trainer.train(train_dataset, test_dataset)

            del train_dataset
            del test_dataset
            del model_trainer

            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
