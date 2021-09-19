from glob import glob

import joblib
import numpy as np
from joblib import Parallel, delayed
from mne import concatenate_raws, pick_types
from scipy.signal import lfilter
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

from config.configuration import Configuration
from src.utils.processing_utils.processing_utils import create_pre_filter, train_feature_extractor, preprocess_data, \
    get_classifier

configuration = Configuration()
# subject = 1
# b, a = create_pre_filter(configuration)
#
# # read and concatenate all the files
# train_raw = concatenate_raws([configuration.load_data(train_file) for train_file in
#                               glob('%ssubj%d_series[1-8]_data.csv' % (configuration.get_dataset_path(), subject))])
#
# # pick eeg signal
# picks = pick_types(train_raw.info, eeg=True)
#
# # Filter data for alpha frequency and beta band
# train_raw._data[picks] = np.array(
#     Parallel(n_jobs=configuration.get_maximum_parallel_jobs())(
#         delayed(lfilter)(b, a, train_raw._data[i]) for i in picks))
#
# # Train feature extractor
# (feature_extractor, extractor_file) = train_feature_extractor(train_raw,
#                                                               picks,
#                                                               configuration)
#
# # Preprocess training data
# data, labels = preprocess_data(train_raw, picks, configuration, subject, list(range(1, 9)),
#                                trained_feature_extractor=feature_extractor)
# joblib.dump((data, labels), 'data_labels_all_sub1.sav')
# del train_raw

data, labels = joblib.load('data_labels_all_sub1.sav')
# iris = datasets.load_iris()
# data = iris.data[:, :2]
# labels = iris.target
classifier, grid = get_classifier(configuration)
scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
search = GridSearchCV(classifier, grid, scoring=scorer, n_jobs=30, verbose=3)
scores = []
for i in range(0, 6):
    print(configuration.get_events()[i])
    results = search.fit(data[:, :].T, np.transpose(labels[i, :]))
    scores.append({
        'accuracy': results.best_score_,
        'config': results.best_params_
    })
best = 0
b_index = None
for index in range(0, len(scores)):
    curr_acc = scores[index]['accuracy']
    if b_index is None or curr_acc > best:
        best = curr_acc
        b_index = index
print('Mean Accuracy: %.3f' % scores[b_index]['accuracy'])
print('Config: %s' % scores[b_index]['config'])
