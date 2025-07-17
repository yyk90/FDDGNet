import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
plt.rcParams['font.family'] = 'SimSun'

def getChannelName( ):
    channel_name = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6','F4', 'F8', 'AF4']
    return channel_name

def createEEG(data,channel_name,sfq):
    info = mne.create_info(
        ch_names=channel_name,
        ch_types=['eeg' for _ in range(len(channel_name))],
        sfreq=sfq
    )
    raw = mne.io.RawArray(data, info)
    return raw


def positioningElectrode(raw):
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    return raw


def averageReference(raw):
    raw.load_data()
    raw.set_eeg_reference(ref_channels='average')
    return raw


def bandpassFilter(raw):
    raw = raw.filter(l_freq=4, h_freq=40)
    raw = raw.notch_filter(freqs=50)
    return raw


def interpolationHighNoiseBadDerivation(raw, person, trial):
    raw.compute_psd().plot()
    scalings = {'eeg': 100}
    if (isinstance(trial,int)):
        raw.plot(n_channels=14, scalings=scalings,
                title='subject：' + person + ' trail：' + str(trial + 1) + 'Please check and select the high-noise bad conductors')
    else:
        if (isinstance(trial, int)):
            raw.plot(n_channels=14, scalings=scalings,
                     title='subject：' + person + ' trail：' + trial + 'Please check and select the high-noise bad conductors')
    plt.show(block=True)
    badflag = False
    if raw.info['bads']:
        print('Bad guide has been selected：', raw.info['bads'], 'Start interpolating bad leads')
        badflag = True
    else:
        print('No bad guidance, skip interpolation')
    if badflag:
        raw.load_data()
        raw.interpolate_bads()
        scalings = {'eeg': 100}
        if (isinstance(trial, int)):
            raw.plot(n_channels=14, scalings=scalings, block=True,
                    title='subject：' + person + ' trail：' + str(trial + 1) + ' Bad guide interpolation completed')
        else:
            raw.plot(n_channels=14, scalings=scalings, block=True,
                     title='subject：' + person + ' trail：' + trial + ' Bad guide interpolation completed')
        plt.show()
    return raw


def icaWithMARA(raw, person, trial):
    scalings = {'eeg': 100}
    ica = ICA(n_components=14, random_state=97)
    ica.fit(raw)

    if (isinstance(trial, int)):
        ica.plot_sources(raw,title='subject：' + person + ' trail：' + str(trial + 1) + ' ICA artifact removal')
    else:
        ica.plot_sources(raw, title='subject：' + person + ' trail：' + trial + ' ICA artifact removal')

    if (isinstance(trial, int)):
        ica.plot_components(title='subject：' + person + ' trail：' + str(trial + 1) + ' Scalp distribution of independent components')
    else:
        ica.plot_components(title='subject：' + person + ' trail：' + trial + ' Scalp distribution of independent components')

    reconst_raw = raw.copy()
    ica.apply(reconst_raw)
    reconst_raw.plot(scalings=scalings, block=True,title='Data after removal')

    return reconst_raw


def interpolationLowAmplitudeBadDerivative(raw, person, trial):
    raw.compute_psd().plot()
    scalings = {'eeg': 100}
    if (isinstance(trial, int)):
        raw.plot(n_channels=14, scalings=scalings,
                 title='subject：' + person + ' trail：' + str(trial + 1) + 'Please check and select the low-noise bad conductors')
    else:
        raw.plot(n_channels=14, scalings=scalings,
                 title='subject：' + person + ' trail：' + trial + 'Please check and select the low-noise bad conductors')
    plt.show(block=True)
    badflag = False
    if raw.info['bads']:
        print('Bad guide has been selected：', raw.info['bads'], 'Start interpolating bad leads')
        badflag = True
    else:
        print('No bad guidance, skip interpolation')
    if badflag:
        raw.load_data()
        raw.interpolate_bads()
        scalings = {'eeg': 100}
        if (isinstance(trial, int)):
            raw.plot(n_channels=14, scalings=scalings, block=True,
                     title='subject：' + person + ' trail：' + str(trial + 1) + ' Bad guide interpolation completed')
        else:
            raw.plot(n_channels=14, scalings=scalings, block=True,
                     title='subject：' + person + ' trail：' + trial + ' Bad guide interpolation completed')
        plt.show()
    return raw

