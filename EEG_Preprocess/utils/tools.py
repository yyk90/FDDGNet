"""
Toolkit for EEG preprocessing process
Including: dataset information, loading EEG data, and various processes
@author YmCui
"""
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import numpy as np
import mne
from mne.preprocessing import ICA

plt.rcParams['font.family'] = 'SimSun'  # 使用宋体作为全局字体


def getChannelName(dataset_name):
    """
    获取通道名称，用于创建脑电数据
    :param dataset_name: 数据集名称
    :return:
    """
    channel_name = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6','F4', 'F8', 'AF4']

    return channel_name

def createEEG(data,channel_name,sfq,dataset):
    """
    创建脑电数据
    :param data: 脑电数据
    channel_name: 通道名称
    sfq: 采样率
    :return: mne标准化的脑电数据
    """
    #['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6','F4', 'F8', 'AF4']
    info = mne.create_info(
        ch_names=channel_name,
        ch_types=['eeg' for _ in range(len(channel_name))],
        sfreq=sfq
    )
    raw = mne.io.RawArray(data, info)
    return raw


def positioningElectrode(raw,dataset):
    """
    定位电极，采用国标10-20模板定位
    :param raw:未经定位的电极
    :return: 定位后的电极
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    return raw


def averageReference(raw,dataset):
    """
    平均参考过程
    :param raw:未经平均参考的数据
    :return:平均参考后的数据
    """
    raw.load_data()
    raw.set_eeg_reference(ref_channels='average')
    return raw


def bandpassFilter(raw,dataset):
    """
    滤波过程，采用4~40HZ带通滤波
    :param raw:未经滤波的数据
    :return: 滤波后的数据
    """
    raw = raw.filter(l_freq=4, h_freq=40)
    raw = raw.notch_filter(freqs=50)
    return raw


def interpolationHighNoiseBadDerivation(raw, person, trial):
    """
    对噪声高的导联进行插值坏导操作
    :param raw: 未经插值的EEG数据
    :param person: 被试编号
    :param trial: 试次编号
    :return: 插值后的EEG数据
    """
    raw.compute_psd().plot()
    scalings = {'eeg': 100}
    if (isinstance(trial,int)):
        raw.plot(n_channels=14, scalings=scalings,
                title='被试编号：' + person + ' 试次：' + str(trial + 1) + '请检查并选中高噪声坏导')
    else:
        if (isinstance(trial, int)):
            raw.plot(n_channels=14, scalings=scalings,
                     title='被试编号：' + person + ' 试次：' + trial + '请检查并选中高噪声坏导')
    plt.show(block=True)
    badflag = False
    if raw.info['bads']:
        print('已选择坏导：', raw.info['bads'], '开始插值坏导')
        badflag = True
    else:
        print('无坏导，跳过插值')
    if badflag:
        raw.load_data()
        raw.interpolate_bads()
        scalings = {'eeg': 100}
        if (isinstance(trial, int)):
            raw.plot(n_channels=14, scalings=scalings, block=True,
                    title='被试编号：' + person + ' 试次：' + str(trial + 1) + ' 坏导插值完成')
        else:
            raw.plot(n_channels=14, scalings=scalings, block=True,
                     title='被试编号：' + person + ' 试次：' + trial + ' 坏导插值完成')
        plt.show()
    return raw


def icaWithMARA(raw, person, trial,dataset_name):
    """
    运用MARA插件进行ICA去伪迹   改一下
    :param person: 被试编号
    :param trial: 试次编号
    :param raw: 未去伪迹的EEG数据
    :return: 去除伪迹的EEG数据
    """
    scalings = {'eeg': 100}
    if dataset_name=='amigos' or dataset_name=='dreamer':
        ica = ICA(n_components=14, random_state=97)
    else:
        ica = ICA(n_components=26, random_state=97)
    ica.fit(raw)
    # 可视化原数据，用以去除伪迹
    if (isinstance(trial, int)):
        ica.plot_sources(raw,title='被试编号：' + person + ' 试次：' + str(trial + 1) + ' ICA去除伪迹')
    else:
        ica.plot_sources(raw, title='被试编号：' + person + ' 试次：' + trial + ' ICA去除伪迹')
    # 画出每一个独立成分的头皮分布
    if (isinstance(trial, int)):
        ica.plot_components(title='被试编号：' + person + ' 试次：' + str(trial + 1) + ' 独立成分的头皮分布')
    else:
        ica.plot_components(title='被试编号：' + person + ' 试次：' + trial + ' 独立成分的头皮分布')

    # 绘制出每个独立成分的一些属性，这里画的太多了，先注释掉
    # pick_id = [i for i in range(30)]
    # ica.plot_properties(raw_notch,pick_id)

    # 手动选择眼电，心电伪影
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)
    reconst_raw.plot(scalings=scalings, block=True,title='去除之后的数据')

    return reconst_raw


def interpolationLowAmplitudeBadDerivative(raw, person, trial):
    """
    对幅值能量低的导联进行插值坏导操作
    :param raw:未经插值的EEG数据
    :param person:被试编号
    :param trial:
    :return:插值后的EEG数据
    """
    raw.compute_psd().plot()
    scalings = {'eeg': 100}
    if (isinstance(trial, int)):
        raw.plot(n_channels=14, scalings=scalings,
                 title='被试编号：' + person + ' 试次：' + str(trial + 1) + '请检查并选中幅值能量低的坏导')
    else:
        raw.plot(n_channels=14, scalings=scalings,
                 title='被试编号：' + person + ' 试次：' + trial + '请检查并选中幅值能量低的坏导')
    plt.show(block=True)
    badflag = False
    if raw.info['bads']:
        print('已选择坏导：', raw.info['bads'], '开始插值坏导')
        badflag = True
    else:
        print('无坏导，跳过插值')
    if badflag:
        raw.load_data()
        raw.interpolate_bads()
        scalings = {'eeg': 100}
        if (isinstance(trial, int)):
            raw.plot(n_channels=14, scalings=scalings, block=True,
                     title='被试编号：' + person + ' 试次：' + str(trial + 1) + ' 坏导插值完成')
        else:
            raw.plot(n_channels=14, scalings=scalings, block=True,
                     title='被试编号：' + person + ' 试次：' + trial + ' 坏导插值完成')
        plt.show()
    return raw


def drawEEG(raw, person, trial, strs = ''):
    """
    绘制脑电图
    :param raw: 待绘制的脑电图
    :param person: 对应的被试编号
    :param trial: 对应的试次
    """
    scalings = {'eeg': 100}
    if (isinstance(trial, int)):
        raw.plot(n_channels = 14, scalings = scalings,
                title = '被试编号：' + person + ' 试次：' + str(trial) + strs)
    else:
        raw.plot(n_channels=14, scalings=scalings,
                 title='被试编号：' + person + ' 试次：' + trial + strs)
    raw.compute_psd().plot()
    plt.show(block = True)
