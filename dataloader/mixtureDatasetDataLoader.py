import csv
import numpy as np
from Path import Path
from torch.utils.data import Dataset
from util.LMDB import LMDBEEGSignalIO

class MixtureDatasetDataLoader(Dataset):

    def __init__(self, current_subject, train_or_test, type, file_path):
        self.info_list = None
        self.current_subject = current_subject
        self.subject_dataset_mapping = {}
        if train_or_test == 'train':
            if type == 'los':
                leave_one_subject_file = file_path + '/los/train_' + str(self.current_subject) + '.csv'
            else:
                raise ValueError(f"Invalid command: {type}")
        elif train_or_test == 'test':
            if type == 'los':
                leave_one_subject_file = file_path + '/los/test_' + str(self.current_subject) + '.csv'
            else:
                raise ValueError(f"Invalid command: {type}")
        else:
            raise ValueError(f"Invalid command: {train_or_test}")
        self.info_list = self.read_csv_file(leave_one_subject_file)

        self.info_list = [item for item in self.read_csv_file(leave_one_subject_file) if int(item[7]) != 999]

        new_id = 0

        for row in self.info_list:
            subject_id = row[3]
            dataset_id = row[8]

            key = (subject_id, dataset_id)

            if key not in self.subject_dataset_mapping:
                self.subject_dataset_mapping[key] = new_id
                new_id += 1

    @staticmethod
    def read_csv_file(path):
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            result = list(reader)
        return result

    def __getitem__(self, index):
        info = self.info_list[index]
        clip_id = info[2]
        subject_id = info[3]
        dataset_id = int(info[8])

        dataset_path = None
        if dataset_id == 0:
            dataset_path = Path.DREAMER_SLICED_DATA_DIR
        elif dataset_id == 1:
            dataset_path = Path.AMIGOS_SLICED_DATA_DIR
        database_file = dataset_path + '/Subject_' + subject_id

        eeg = LMDBEEGSignalIO(database_file).read_eeg(clip_id)
        eeg = np.array(eeg)
        eeg = np.transpose(eeg)

        trial_id = int(info[4])
        valence = float(info[5])
        arousal = float(info[6])
        label = int(info[7])

        key = (subject_id, str(dataset_id))
        subject_id_new = self.subject_dataset_mapping.get(key)

        return eeg, label, subject_id_new, trial_id, valence, arousal

    def __len__(self):
        return len(self.info_list) - 1
