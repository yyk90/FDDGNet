import scipy
from EEG_Preprocess.tools import positioningElectrode, averageReference, bandpassFilter, interpolationLowAmplitudeBadDerivative, createEEG, getChannelName
from EEG_Preprocess.tools import interpolationHighNoiseBadDerivation, icaWithMARA
import scipy.io as sio
import os
from Path import Path

# amigos  or  dreamer
dataset_name = 'amigos'

folder_path = Path.PROJECT_DIR + '/data_mat/' + dataset_name

trail = 16

sfreq = 128

try:
    all_mat_file = os.walk(folder_path)
    skip_set = {}
    channel_name = getChannelName()

    for path, dir_list, file_list in all_mat_file:
        for file_name in file_list:
            if file_name not in skip_set:

                mat_data = sio.loadmat(folder_path + "/" + file_name)

                subjectNumber = file_name

                # The format of the file name must be Subject_id ( id starts from 1 )

                savePath = Path.PROJECT_DIR + "/Preprocessed_Data/"+str.upper(dataset_name)+"/" + subjectNumber

                pre_Data = {}
                data = []

                for i in range(trail):
                    if mat_data['EEG_DATA'][0, i].size>0:
                        data = list(mat_data['EEG_DATA'][0, i].T[slice(3, 17, 1)])

                    if data != []:

                        raw = createEEG(data,channel_name,sfreq)

                        raw = positioningElectrode(raw)

                        raw = bandpassFilter(raw)

                        interpolationHighNoiseBadDerivation(raw,subjectNumber,i)

                        raw = icaWithMARA(raw,subjectNumber,i)

                        interpolationLowAmplitudeBadDerivative(raw,subjectNumber,i)

                        raw = averageReference(raw)

                        preprocesed_data, times = raw[:]

                        pre_Data[str(i+1)] = preprocesed_data

                    else:
                        pre_Data[str(i+1)] = []
                        with open(Path.PROJECT_DIR + "/" + dataset_name + "_null_list.txt","a") as f:
                            f.write(subjectNumber + "   " + str(i+1) + "is null [ ]\n")

                    print(subjectNumber + "   " +str(i+1))

                scipy.io.savemat(savePath, pre_Data , appendmat=True)

except FileNotFoundError as e:
    print('An error occurred while loading data: {}'.format(e))