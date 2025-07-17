"""
路径文件，管理项目中使用的全部路径。
"""


class Path:
    """
    路径类，管理项目中使用的全部绝对路径。
    """
    # NOTE: 在新机器上运行此程序时，修改以下路径即可令程序正常运行
    PROJECT_DIR = '/home/uais3/dlf/IT_DG'
    # 预处理后dreamer数据集所在路径
    DREAMER_P_EEG_PATH = PROJECT_DIR + "/Preprocessed_Data/DREAMER"
    # 预处理后amigos数据集所在路径
    AMIGOS_P_EEG_PATH = PROJECT_DIR + "/Preprocessed_Data/AMIGOS"


    # dreamer数据集切片所在目录
    DREAMER_SLICED_DATA_DIR = PROJECT_DIR + "/sliceddata/dreamer"
    # amigos数据集切片所在目录
    AMIGOS_SLICED_DATA_DIR = PROJECT_DIR + "/sliceddata/amigos"

    # dreamer数据集标签文件路径
    DREAMER_LABELS = DREAMER_P_EEG_PATH + "/DreamerLabel.xls"
    # amigos数据集标签文件路径
    AMIGOS_LABELS = AMIGOS_P_EEG_PATH + "/AmigosLabel.xls"

    # 数据集训练信息表所在目录
    FORMAT_DIR = PROJECT_DIR + "/format/"

