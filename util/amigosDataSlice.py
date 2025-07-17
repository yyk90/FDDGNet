import re
import os
import csv
import xlrd
import scipy.io as scio
from Path import Path
from util.LMDB import LMDBEEGSignalIO


class AmigosDataSlice:
    @staticmethod
    def load_data(file_path,seq_len):
        positive = 0
        negative = 0
        invalid = 0

        os.makedirs(Path.AMIGOS_SLICED_DATA_DIR)
        label = AmigosDataSlice.load_labels()

        all_mat_file = os.walk(file_path)
        skip_set = {'AmigosLabel.xls'}
        skip_subject = { }
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                if file_name not in skip_set:

                    subject_id = ''.join(re.findall(r'\d+', file_name.split("_")[1]))

                    if subject_id not in skip_subject:
                        dir_path = Path.AMIGOS_SLICED_DATA_DIR + '/Subject_' + subject_id

                        database = LMDBEEGSignalIO(dir_path)

                        header = ['start_at', 'end_at', 'clip_id', 'subject_id', 'trial_id',
                                  'valence', 'arousal', 'label', 'dataset']
                        with open(dir_path+'/msg.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(header)

                        subject_data = scio.loadmat(file_path + "/" + file_name)

                        for i in range(16):
                            subject_eeg_trail = subject_data['time' + str(i + 1)]

                            arousal = label[(int(subject_id) - 1) * 16 + i][1]
                            valence = label[(int(subject_id) - 1) * 16 + i][2]
                            quality = label[(int(subject_id) - 1) * 16 + i][3]
                            valence_temp = float(valence)
                            quality = int(quality)

                            if quality == 0:
                                if valence_temp >= 5.0:
                                    discrete_label = 1
                                    positive = positive + 1
                                elif valence_temp < 5.0:
                                    discrete_label = 0
                                    negative = negative + 1
                                else:
                                    discrete_label = 999
                                    invalid = invalid + 1
                            else:
                                discrete_label = 999

                            start_at = 0

                            while start_at + seq_len < subject_eeg_trail.shape[1]:

                                end_at = start_at + seq_len
                                second_eeg = subject_eeg_trail[:, start_at:end_at]

                                clip_id = database.write_eeg(second_eeg)

                                data = [start_at, end_at, clip_id, subject_id, i, valence, arousal, discrete_label, 1]
                                with open(dir_path + '/msg.csv', 'a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow(data)
                                start_at = end_at
                            print("In the amigos data slice... Please wait ...")

        print("Various sample sizes (positive, negative, invalid)：")
        print(positive)
        print(negative)
        print(invalid)

    @staticmethod
    def load_labels():
        label = []
        file_path = Path.AMIGOS_LABELS
        workbook = xlrd.open_workbook(file_path)
        sheet = workbook.sheet_by_index(0)

        for row_idx in range(2, sheet.nrows):
            row = sheet.row_values(row_idx)

            if not all(cell == '' for cell in row):
                label.append(row)
        return label

    @staticmethod
    def read_csv_file(path):
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            result = list(reader)
        return result
