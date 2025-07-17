import scipy

from utils.tools import positioningElectrode, averageReference, bandpassFilter, interpolationLowAmplitudeBadDerivative, \
    drawEEG, createEEG, getChannelName
from utils.tools import interpolationHighNoiseBadDerivation, icaWithMARA
import scipy.io as sio
import os
import numpy as np

# 数据集名称：【文件地址，被试看多少段视频，采样率，mat文件数据起始下标】
DataSet = {
    'amigos':['F:\OpenDataSet\AMIGOS dataset\data\data_original\data_mat',20,128,0]
}

# 数据集名称
dataset_name = 'amigos'

# 文件路径
folder_path = DataSet[dataset_name][0]

# 每个被试看多少段视频
trail = DataSet[dataset_name][1]
# 采样率
sfreq = DataSet[dataset_name][2]
# 读出的mat格式开始存储数据的下标
start_idx = DataSet[dataset_name][3]

try:
    all_mat_file = os.walk(folder_path)  # 访问文件夹里全部的文件
    skip_set = {}
    channel_name = getChannelName(dataset_name)
    epoch = 0
    # 访问所有的文件
    for path, dir_list, file_list in all_mat_file:
        for file_name in file_list:
            epoch+=1
            if file_name not in skip_set:

                # 得到当前文件中的数据
                data = sio.loadmat(folder_path + "/" + file_name)
                # # 从文件名中获取被试编号。直接拿文件名当被试编号
                subjectNumber = file_name
                # 保存数据的文件地址
                if dataset_name=='amigos':
                    savePath = "F:\OpenDataSet\AMIGOS dataset\data\My_preprocessed_EEG\\" + subjectNumber
                # 保存数据的字典
                if dataset_name=='amigos':
                    pre_Data = {}

                # 每个被试看trail段视频
                for i in range(trail):
                    # 提取本段视频的脑电数据
                    if dataset_name=='amigos':
                        #print(data['EEG_DATA'][0, i])
                        if data['EEG_DATA'][0, i].size>0:
                            data1 = list(data['EEG_DATA'][0, i].T[slice(3, 17, 1)])
                        else:
                            data1 = []
                    # 截取前14行有用的
                    #data2 = data1[:14]
                    if data1!=[]:

                        # 创建脑电数据
                        raw = createEEG(data1,channel_name,sfreq,dataset_name)
                        # 定位电极
                        raw = positioningElectrode(raw,dataset_name)
                        # 带通滤波
                        raw = bandpassFilter(raw,dataset_name)
                        # 插值坏导 针对噪声高的导联
                        interpolationHighNoiseBadDerivation(raw,subjectNumber,i)
                        # ICA去除伪迹
                        raw = icaWithMARA(raw,subjectNumber,i,dataset_name)
                        # 插值坏导 针对幅值能量低的导联
                        interpolationLowAmplitudeBadDerivative(raw,subjectNumber,i)
                        # 平均参考过程
                        raw = averageReference(raw,dataset_name)
                        # 绘制脑电数据
                        drawEEG(raw, subjectNumber, i + 1,'处理完之后的数据')
                        #获取到处理完之后的数据,数据格式为 numpy.ndarray
                        preprocesed_data, times = raw[:]
                        # 保存数据
                        # 获取变量名称
                        if dataset_name == 'amigos':
                            pre_Data[str(i+1)] = preprocesed_data

                    else:
                        if dataset_name == 'amigos':
                            pre_Data[str(i+1)] = []
                            with open("C:\\Users\\admin\Desktop\\amigos截图\\none.txt","a") as f:
                                f.write(subjectNumber + "   " +str(i+1)+"为空 [ ]\n")

                    print(subjectNumber + "   " +str(i+1))
                if dataset_name=='amigos':
                    scipy.io.savemat(savePath, pre_Data , appendmat=True)

except FileNotFoundError as e:
    print('加载数据时出错: {}'.format(e))