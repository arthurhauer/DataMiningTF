import csv
import pathlib
import statistics
from glob import glob
from os.path import exists

import numpy
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
from src.utils.utils import cross_validation_prepare, creat_mne_raw_object, extract_events_from_file, DTWDistance

# def split_by_subject(file_name) -> list:
#     file = open(file_name)
#     csvf = csv.reader(file)
#     next(csvf)
#     subjects = [[], [], [], [], [], [], [], [], [], [], [], []]
#     print('Reading CSV files')
#     for row in csvf:
#         subject = int(float(row[20])) - 1
#         values = list(map(float, row[0:-1]))
#         if len(subjects[subject]) > 1225:
#             continue
#         subjects[subject].append(values)
#     return subjects
#
#
# files = glob('../extracted_data/*.csv')
# config = Configuration()
#
# for fname in files:
#     for fname2 in files:
#         subjects_1 = split_by_subject(fname)
#         subjects_2 = split_by_subject(fname2)
#         move_sub_1 = str(fname).replace('../extracted_data\\', '').replace('.csv', '')
#         move_sub_2 = str(fname2).replace('../extracted_data\\', '').replace('.csv', '')
#         for sub1 in range(len(subjects_1)):
#             results = []
#             other_subs = list(range(len(subjects_1)))
#             del other_subs[sub1]
#             for sub2 in other_subs:
#                 print('Beginning DWT on %s subject %d and %s subject %d' % (move_sub_1, sub1 + 1, move_sub_2, sub2 + 1))
#                 distance = DTWDistance(subjects_1[sub1], subjects_2[sub2], 10)
#                 print('Distance between %s subject %d and %s subject %d = %f' % (
#                     move_sub_1, sub1 + 1, move_sub_2, sub2 + 1, distance))
#                 results.append([move_sub_1, move_sub_2, move_sub_1 == move_sub_2, sub1 + 1, sub2 + 1, distance])
#             config.save_result(results, ['Movement1', 'Movement2', 'SameClass', 'Subject1', 'Subject2', 'Distance'])

file = open('../results/results.csv')
csvf = csv.reader(file)
next(csvf)
values = {
    'same_class': [],
    'different_classes': []
}
for row in csvf:
    values['same_class' if row[2] == 'True' else 'different_classes'].append(float(row[5]))

print('Same class mean distance: %f' % statistics.mean(values['same_class']))
print('Same class std deviation: %f' % statistics.stdev(values['same_class']))
print('Different classes mean distance: %f' % statistics.mean(values['different_classes']))
print('Different classes std deviation: %f' % statistics.stdev(values['different_classes']))


# def split_by_label(train_data, labels, subject, configuration_data: Configuration):
#     path = '../extracted_data/%s.csv'
#     files = []
#     writers = []
#     new_data = []
#     events = configuration_data.get_events()
#     for label in events:
#         result_path = path % label
#         if not exists(result_path):
#             pathlib.Path.touch(pathlib.Path(result_path))
#             file = open(result_path, 'a', newline='\n', encoding='utf-8')
#             writer = csv.writer(file)
#             headers = ['Feature%d' % feature_index for feature_index in range(1, train_data.shape[0] + 1)]
#             headers.append('Subject')
#             writer.writerow(headers)
#             file.close()
#         file = open(result_path, 'a', newline='\n', encoding='utf-8')
#         files.append(file)
#         writers.append(csv.writer(file))
#     subject_column = np.ones((1, train_data.shape[1])) * subject
#     train_data = np.vstack([train_data, subject_column])
#     eye = np.eye(6)
#     n_events = len(events)
#     for new_data_index in range(0, n_events):
#         new_data.append([])
#     for index in range(1, train_data.shape[1]):
#         for new_data_index in range(0, n_events):
#             if (np.array_equal(labels[0:n_events, index], eye[:, new_data_index])):
#                 new_data[new_data_index].append(train_data[:, index])
#                 break
#     for index in range(0, n_events):
#         writers[index].writerows(new_data[index][:])
#         files[index].close()
#
#
# configuration = Configuration()
# b, a = create_pre_filter(configuration)
# for subject in range(1, 13):
#     train_files = glob(configuration.get_dataset_path() + 'subj%d_series[1-8]_data.csv' % subject)
#     train_raw = concatenate_raws([configuration.load_data(train_file) for train_file in train_files])
#
#     # pick eeg signal
#     picks = pick_types(train_raw.info, eeg=True)
#
#     # Filter data for alpha frequency and beta band
#     train_raw._data[picks] = np.array(
#         Parallel(n_jobs=configuration.get_maximum_parallel_jobs())(
#             delayed(lfilter)(b, a, train_raw._data[i]) for i in picks))
#
#     # Train feature extractor
#     (feature_extractor, extractor_file) = train_feature_extractor(train_raw,
#                                                                   picks,
#                                                                   configuration)
#
#     # Preprocess training data
#     training_data, labels = preprocess_data(train_raw, picks, configuration, subject, list(range(1, 9)),
#                                             trained_feature_extractor=feature_extractor)
#     del train_raw
#
#     split_by_label(training_data, labels, subject, configuration)
