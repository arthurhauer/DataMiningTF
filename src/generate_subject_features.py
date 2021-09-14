import csv
import pathlib
from glob import glob
from os.path import exists
import numpy as np
import joblib

from config.configuration import Configuration
import time
import pandas as pd
import uuid
from mne import concatenate_raws, pick_types

from scipy.signal import lfilter

from sklearn import metrics

from src.utils.processing_utils.processing_utils import *
from src.utils.utils import cross_validation_prepare, creat_mne_raw_object, extract_events_from_file


def split_by_label(train_data, labels, subject, configuration_data: Configuration):
    path = '../extracted_data/%s.csv'
    files = []
    writers = []
    new_data = []
    events = configuration_data.get_events()
    for label in events:
        result_path = path % label
        if not exists(result_path):
            pathlib.Path.touch(pathlib.Path(result_path))
            file = open(result_path, 'a', newline='\n', encoding='utf-8')
            writer = csv.writer(file)
            headers = ['Feature%d' % feature_index for feature_index in range(1, train_data.shape[0] + 1)]
            headers.append('Subject')
            writer.writerow(headers)
            file.close()
        file = open(result_path, 'a', newline='\n', encoding='utf-8')
        files.append(file)
        writers.append(csv.writer(file))
    subject_column = np.ones((1, train_data.shape[1])) * subject
    train_data = np.vstack([train_data, subject_column])
    eye = np.eye(6)
    n_events = len(events)
    for new_data_index in range(0, n_events):
        new_data.append([])
    for index in range(1, train_data.shape[1]):
        for new_data_index in range(0, n_events):
            if (np.array_equal(labels[0:n_events, index], eye[:, new_data_index])):
                new_data[new_data_index].append(train_data[:, index])
                break
    for index in range(0, n_events):
        writers[index].writerows(new_data[index][:])
        files[index].close()


configuration = Configuration()
b, a = create_pre_filter(configuration)
for subject in range(1, 13):
    train_files = glob(configuration.get_dataset_path() + 'subj%d_series[1-8]_data.csv' % subject)
    train_raw = concatenate_raws([configuration.load_data(train_file) for train_file in train_files])

    # pick eeg signal
    picks = pick_types(train_raw.info, eeg=True)

    # Filter data for alpha frequency and beta band
    train_raw._data[picks] = np.array(
        Parallel(n_jobs=configuration.get_maximum_parallel_jobs())(
            delayed(lfilter)(b, a, train_raw._data[i]) for i in picks))

    # Train feature extractor
    (feature_extractor, extractor_file) = train_feature_extractor(train_raw,
                                                                  picks,
                                                                  configuration)

    # Preprocess training data
    training_data, labels = preprocess_data(train_raw, picks, configuration, subject, list(range(1, 9)),
                                            trained_feature_extractor=feature_extractor)
    del train_raw

    split_by_label(training_data, labels, subject, configuration)
