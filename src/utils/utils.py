import gc
from math import sqrt
from glob import glob
from typing import Any

import numpy as np
import pandas as pd
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray


def extract_events_from_file(file) -> Any:
    events_raw = pd.read_csv(file)
    events_names = events_raw.columns[1:]
    events_data = np.array(events_raw[events_names]).T
    return events_data, events_names


def compare_predicted_and_actual(events_data, predicted):
    predicted_event_quadratic_error = np.absolute(events_data - predicted)
    return predicted_event_quadratic_error


def resample_data(gt, chunk_size=1000):
    """
    split long signals to smaller chunks, discard no-events chunks
    """
    total_discard_chunks = 0
    mean_val = []
    threshold = 0.01
    index = []

    for i in range(len(gt)):
        for j in range(0, len(gt[i]), chunk_size):
            mean_val.append(np.mean(gt[i][:, j:min(len(gt[i]), j + chunk_size)]))
            if mean_val[-1] < threshold:  # discard chunks with low events time
                total_discard_chunks += 1
            else:
                index.extend([(i, k) for k in range(j, min(len(gt[i]), j + chunk_size))])

    print('Total number of chunks discarded: {} chunks'.format(total_discard_chunks))
    print('{}% data'.format(total_discard_chunks / len(mean_val)))
    del mean_val
    gc.collect()
    return index


def cross_validation_prepare(iteration, subject_index, dataset_path) -> Any:
    train_series = list(range(1, 9))
    test_series = iteration + 1
    del train_series[iteration]
    train_files = []
    for series in train_series:
        train_files.append(dataset_path + 'subj%d_series%d_data.csv' % (subject_index, series))
    return train_files, train_series, [
        dataset_path + 'subj%d_series%d_data.csv' % (subject_index, test_series)], [test_series]


def creat_mne_raw_object(fname, read_events=True):
    print('Reading EEG data')

    """Create a mne raw instance from csv file"""
    # Read EEG file
    data = pd.read_csv(fname)

    # get chanel names
    ch_names = list(data.columns[1:])

    # read EEG standard montage from mne
    montage = make_standard_montage('standard_1005')

    ch_type = ['eeg'] * len(ch_names)
    data = 1e-6 * np.array(data[ch_names]).T

    if read_events:
        # events file
        ev_fname = fname.replace('_data', '_events')
        # read event file
        events_data, events_names = extract_events_from_file(ev_fname)

        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim'] * 6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data, events_data))

    # create and populate MNE info structure
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type)

    # create raw object
    raw_array = RawArray(data, info, verbose=False)
    raw_array.set_montage(montage)

    return raw_array


def DTWDistance(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))
    print('Initializing distances matrix')
    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    print('Calculating distances')
    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = []
            for k in range(len(s1[0])):
                dist.append((s1[i][k] - s2[j][k]) ** 2)
            DTW[(i, j)] = np.average(dist) + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])
