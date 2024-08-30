import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from py_neuromodulation import nm_stream, nm_define_nmchannels, nm_settings, nm_analysis
from joblib import Parallel, delayed


def read_data(PATH_DAT):
    b = open(PATH_DAT, "rb").read()
    ecog = np.frombuffer(b, dtype=np.int16)
    ecog = ecog - 512
    ecog = ecog.reshape([-1, 4])
    return ecog.T


def compute_features(
    PATH_DAT,
    dat_file,
    subject,
):
    ecog = read_data(PATH_DAT + "/" + dat_file)

    fs = 250

    t = np.arange(0, ecog.shape[1]) / fs

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(
        ecog, car_rereferencing=False
    )
    settings = nm_settings.NMSettings()
    settings_rns = settings.get_fast_compute()
    settings_rns.postprocessing["feature_normalization"] = False
    settings_rns.features.bispectrum = True
    settings_rns.features.sharpwave_analysis = True
    settings_rns.sharpwave_analysis_settings.estimator.var.append("interval")
    settings_rns.sharpwave_analysis_settings.estimator.var.append("width")
    settings_rns.sharpwave_analysis_settings.sharpwave_features.num_peaks = True
    settings_rns.sharpwave_analysis_settings.estimator.mean.append("num_peaks")
    settings_rns.frequency_ranges_hz.pop("HFA")
    settings_rns.frequency_ranges_hz.pop("high_gamma")

    settings_rns.segment_length_features_ms = 10000

    stream = nm_stream.Stream(
        sfreq=fs,
        nm_channels=nm_channels,
        sampling_rate_features_hz=1 / 10,
        settings=settings_rns,
        line_noise=60,
    )

    features = stream.run(
        data=ecog.astype(float),
        out_path_root="features/" + subject,
        folder_name=dat_file[:-4],
    )

    # e.g. ch3_avgref_Sharpwave_Mean_num_peaks_range_5_30 is not a column


if __name__ == "__main__":
    subjects = os.listdir("selected_dats")
    subject = subjects[0]
    PATH_DAT = "selected_dats/" + subject
    dat_files = os.listdir(PATH_DAT)
    dat_file = dat_files[29]

    compute_features(PATH_DAT, dat_file, subject)
    print("Number of cores: ", os.cpu_count())

    Parallel(n_jobs=-1)(
        delayed(compute_features)(PATH_DAT, dat_file, subject) for dat_file in dat_files
    )
