import csv
from Path import Path

class GenerateFormat:

    def __init__(self, datasets, los_file_path):
        valid_datasets = ['dreamer', 'amigos']

        for dataset in datasets:
            if dataset not in valid_datasets:
                raise ValueError(f"Invalid dataset: {dataset}")

        self.datasets = datasets

        self.datasets_msg_dict = {}

        self.los_file_path = los_file_path

    def create_temporary_dict(self):
        index = 1
        for dataset in self.datasets:
            if dataset == 'dreamer':
                for i in range(1, 24):
                    subject_msg_list = []
                    file_path = Path.DREAMER_SLICED_DATA_DIR + '/Subject_' + str(i) + '/msg.csv'

                    with open(file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)

                        next(reader)

                        for row in reader:
                            subject_msg_list.append(row)
                        self.datasets_msg_dict[index] = subject_msg_list
                        index += 1
            if dataset == 'amigos':
                for i in range(1, 41):
                    subject_msg_list = []
                    if i in [11, 16, 23, 24, 32, 33]:
                        continue
                    file_path = Path.AMIGOS_SLICED_DATA_DIR + '/Subject_' + str(i) + '/msg.csv'

                    with open(file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)

                        next(reader)

                        for row in reader:
                            subject_msg_list.append(row)
                        self.datasets_msg_dict[index] = subject_msg_list
                        index += 1

    def generate_leave_one_subject_format(self):
        self.create_temporary_dict()

        for i in self.datasets_msg_dict.keys():
            for key, value in self.datasets_msg_dict.items():
                if key == i:

                    with open(self.los_file_path + '/los/test_' + str(i) + '.csv', 'a', newline='') as file:
                        writer = csv.writer(file)

                        if file.tell() == 0:
                            writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                            'valence', 'arousal', 'label', 'dataset'])
                        for row in value:
                            writer.writerow(row)
                else:

                    with open(self.los_file_path + '/los/train_' + str(i) + '.csv', 'a', newline='') as file:
                        writer = csv.writer(file)

                        if file.tell() == 0:
                            writer.writerow(['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                             'valence', 'arousal', 'label', 'dataset'])
                        for row in value:
                            writer.writerow(row)