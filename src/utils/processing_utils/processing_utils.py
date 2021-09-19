from typing import Any, List, Tuple

import numpy as np
from numpy import arange
from sklearn import svm, neural_network

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.epochs import concatenate_epochs
from mne import find_events, Epochs
from mne.decoding import CSP

from scipy.signal import butter, convolve, get_window

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from joblib import Parallel, delayed
from sklearn.multiclass import OneVsRestClassifier

from config.configuration import Configuration


def create_pre_filter(configuration_data: Configuration) -> Any:
    print('Generating pre filter')
    if configuration_data.get_pre_filtering_type() == 'butterworth':
        frequencies = [
            configuration_data.get_pre_filtering_settings()['minimum-frequency'],
            configuration_data.get_pre_filtering_settings()['maximum-frequency']
        ]
        filter_order = configuration_data.get_pre_filtering_settings()['order']
        return butter(filter_order,
                      np.array(frequencies) / configuration_data.get_sampling_frequency(), btype='bandpass')
    else:
        raise Exception("Unsupported pre-filtering type. Available filters: " + "butterworth")


def train_feature_extractor(data, data_picks, configuration_data: Configuration, extractor=None) -> Any:
    epochs_tot = []
    if configuration_data.has_preloaded_extractor():
        print('Loading extractor file...')
        extractor = configuration_data.get_preloaded_extractor()
        ex_file = configuration_data.get_preload_extractor_file()
    if configuration_data.should_train_extractor() or extractor is None:
        print('Starting feature extractor training')
        event_window = configuration_data.get_event_window()
        y = []

        events_data = find_events(data, stim_channel=configuration_data.get_events(), verbose=False)
        epochs = Epochs(data, events_data, {'during': 1}, 0, event_window, proj=False,
                        picks=data_picks, baseline=None, preload=True,
                        verbose=False)

        epochs_tot.append(epochs)
        y.extend([1] * len(epochs))
        epochs_rest = Epochs(data, events_data, {'before': 1}, -event_window, 0, proj=False,
                             picks=data_picks, baseline=None, preload=True,
                             verbose=False)

        # Workaround to be able to concatenate epochs with MNE
        epochs_rest.set_times(epochs.times)
        y.extend([-1] * len(epochs_rest))
        epochs_tot.append(epochs_rest)
        # Concatenate all epochs
        epochs = concatenate_epochs(epochs_tot)

        # get data
        x = epochs.get_data()
        y = np.array(y)
        extractor_type = configuration_data.get_feature_extractor_type()
        if extractor_type == "csp":
            print('Training CSP')
            num_filters = configuration_data.get_feature_extractor_settings()['number-of-filters']
            regularization = configuration_data.get_feature_extractor_settings()['regularization']
            if num_filters is None:
                num_filters = 4
            if regularization is None:
                regularization = 'ledoit_wolf'
            # train CSP
            if extractor is None:
                extractor = CSP(n_components=num_filters, reg=regularization)
            extractor.fit(x, y)

        else:
            raise Exception('Unsupported feature extractor. Available types: ' + 'csp')
        ex_file = configuration_data.save_extractor(extractor)
    print('Done!')
    return extractor, ex_file


def preprocess_data(data, data_picks, configuration_data: Configuration, subject_index: int, series_list: List[int],
                    trained_feature_extractor: Any = None) -> Any:
    print('Starting data pre-process...')
    extracted_data: Any
    processed_data: Any
    number_of_filters = configuration_data.get_feature_extractor_settings()['number-of-filters']
    smoothing_type = configuration_data.get_smoothing_type()
    if smoothing_type is None:
        smoothing = None
    else:
        smoothing = get_window(smoothing_type, configuration_data.get_smoothing_window_size())

    print('Extracting features')
    if trained_feature_extractor is None:
        extracted_data = data
    elif isinstance(trained_feature_extractor, CSP):
        extracted_data = np.dot(trained_feature_extractor.filters_[0:number_of_filters], data._data[data_picks]) ** 2
    else:
        raise Exception("Unsupported feature extractor. Available types: " + 'csp')
    if smoothing is not None:
        print('Smoothing')
        processed_data = np.array(
            Parallel(n_jobs=configuration_data.get_maximum_parallel_jobs())(
                delayed(convolve)(extracted_data[i], smoothing, 'full') for i in range(number_of_filters)))
    else:
        processed_data = extracted_data

    processed_data = np.log(processed_data[:, 0:extracted_data.shape[1]])
    processed_data = np.asarray(processed_data, dtype=np.float32)
    label_data = np.asarray(data._data[32:], dtype=np.float32)
    configuration_data.save_extracted_data(processed_data, label_data, subject_index, series_list)
    return processed_data, label_data


def get_classifier(configuration_data: Configuration) -> Tuple[Any, Any]:
    print('Choosing classifier')
    if configuration_data.has_preloaded_model():
        return configuration_data.get_preloaded_model()
    else:
        chosen_classifier = configuration_data.get_classifier_type()
        if chosen_classifier == "multi-layer-perceptron":
            return neural_network.MLPClassifier(
                max_iter=500,
                learning_rate='adaptive',
                hidden_layer_sizes=(100, 100, 75, 50, 25)
            ), {}
        elif chosen_classifier == 'linear-discriminant-analysis':
            return LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.1), [
                {
                    'solver': ['lsqr', 'eigen'],
                    'shrinkage': arange(0, 1, 0.1)
                }
            ]

        elif chosen_classifier == 'logistic-regression':
            return LogisticRegression(

            ), {}
        elif chosen_classifier == 'support-vector-machine':
            return svm.SVC(
                max_iter=500,
                probability=True
            ), {}
        else:
            raise Exception(
                'Unsupported classifier. Available classifiers:' + 'multi-layer-perceptron, ' + 'linear-discriminant-analysis, ' + 'logistic-regression, ' + 'support-vector-machine')


def gridsearch_classifier_tuning(training_data: Any, training_labels: Any, classifier: Any, grid: dict,
                                 configuration: Configuration) -> Any:
    scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True, multi_class='ovr')
    search = GridSearchCV(classifier, grid, scoring=scorer, n_jobs=-1, verbose=3)
    results = search.fit(training_data[:, :].T, np.transpose(training_labels[:, :]))
    classifier.set_params(results.best_params_)
    return classifier
