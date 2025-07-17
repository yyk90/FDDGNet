class Path:
    # NOTE: When running this program on a new machine, simply modify the following path to ensure the program runs normally
    PROJECT_DIR = './code_of_model'

    DREAMER_P_EEG_PATH = PROJECT_DIR + "/Preprocessed_Data/DREAMER"
    AMIGOS_P_EEG_PATH = PROJECT_DIR + "/Preprocessed_Data/AMIGOS"


    DREAMER_SLICED_DATA_DIR = PROJECT_DIR + "/sliceddata/dreamer"
    AMIGOS_SLICED_DATA_DIR = PROJECT_DIR + "/sliceddata/amigos"


    DREAMER_LABELS = DREAMER_P_EEG_PATH + "/DreamerLabel.xls"
    AMIGOS_LABELS = AMIGOS_P_EEG_PATH + "/AmigosLabel.xls"


    FORMAT_DIR = PROJECT_DIR + "/format/"

