import time
from typing import Any

import numpy as np
import pandas as pd
import tsfel
import uuid
from scipy.signal.windows import boxcar
from sklearn import svm, neural_network

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.epochs import concatenate_epochs
from mne import find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from scipy.signal import butter, lfilter, convolve, get_window

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from joblib import Parallel, delayed

from config.configuration import Configuration
from src.utils.utils import cross_validation_prepare, creat_mne_raw_object, extract_events_from_file


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


def train_feature_extractor(data, data_picks, configuration_data: Configuration) -> Any:
    extractor = None
    if configuration_data.has_preloaded_extractor():
        print('Loading extractor file...')
        extractor = configuration_data.get_preloaded_extractor()
        ex_file = configuration_data.get_preload_extractor_file()
    if configuration_data.should_train_extractor() or extractor is None:
        print('Starting feature extractor training')
        event_window_before = configuration_data.get_event_window_before()
        event_window_after = configuration_data.get_event_window_after()
        y = []
        # get event position corresponding to HandStart
        events_data = find_events(data, stim_channel=configuration_data.get_events(), verbose=False)
        # epochs signal for 2 second after the event
        epochs = Epochs(data, events_data, {'during': 1}, 0, event_window_after, proj=False,
                        picks=data_picks, baseline=None, preload=True,
                        verbose=False)

        epochs_tot.append(epochs)
        y.extend([1] * len(epochs))

        # epochs signal for 2 second before the event, this correspond to the
        # rest period.
        epochs_rest = Epochs(data, events_data, {'before': 1}, -event_window_before, 0, proj=False,
                             picks=data_picks, baseline=None, preload=True,
                             verbose=False)

        # Workaround to be able to concatenate epochs with MNE
        epochs_rest.set_times(epochs.times)
        #
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

        elif extractor_type == 'tsfel':
            print('Training TSFEL')
            if extractor is None:
                extractor = tsfel.get_features_by_domain()
        else:
            raise Exception('Unsupported feature extractor. Available types: ' + 'csp, ' + 'tsfel')
        ex_file = configuration_data.save_extractor(extractor)
    print('Done!')
    return extractor, ex_file


def preprocess_data(data, data_picks, configuration_data: Configuration,
                    trained_feature_extractor: Any = None) -> np.ndarray:
    print('Starting data pre-process...')
    extracted_data: Any
    processed_data: Any
    number_of_filters = configuration_data.get_feature_extractor_settings()['number-of-filters']
    smoothing_type = configuration_data.get_smoothing_type()
    if smoothing_type is None:
        smoothing = None
    #
    # elif smoothing_type == 'boxcar':
    #     smoothing = boxcar(configuration_data.get_smoothing_window_size())
    else:
        # raise Exception('Unsupported smoothing type. Available types: ' + 'boxcar')
        smoothing = get_window(smoothing_type, configuration_data.get_smoothing_window_size())

    print('Extracting features')
    if trained_feature_extractor is None:
        extracted_data = data
    elif isinstance(feature_extractor, CSP):
        extracted_data = np.dot(trained_feature_extractor.filters_[0:number_of_filters], data._data[data_picks]) ** 2
    elif configuration_data.get_feature_extractor_type() == 'tsfel':
        extracted_data = tsfel.time_series_features_extractor(trained_feature_extractor, data._data[data_picks],
                                                              fs=configuration_data.get_sampling_frequency())
    else:
        raise Exception("Unsupported feature extractor. Available types: " + 'csp, ' + 'tsfel')
    print('Smoothing')
    if smoothing is not None:
        processed_data = np.array(
            Parallel(n_jobs=configuration_data.get_maximum_paralel_jobs())(
                delayed(convolve)(extracted_data[i], smoothing, 'full') for i in range(number_of_filters)))
    else:
        processed_data = extracted_data

    processed_data = np.log(processed_data[:, 0:extracted_data.shape[1]])
    processed_data = np.asarray(processed_data, dtype=np.float32)
    return processed_data


def get_classifier(configuration_data: Configuration) -> Any:
    print('Choosing classifier')
    if configuration.has_preloaded_model():
        return configuration.get_preloaded_model()
    else:
        chosen_classifier = configuration_data.get_classifier_type()
        if chosen_classifier == "multi-layer-perceptron":
            return neural_network.MLPClassifier(
                learning_rate='adaptive',
                hidden_layer_sizes=(100, 100, 75, 50, 25)
            )
        elif chosen_classifier == 'linear-discriminant-analysis':
            return LinearDiscriminantAnalysis(

            )
        elif chosen_classifier == 'logistic-regression':
            return LogisticRegression(

            )
        elif chosen_classifier == 'support-vector-machine':
            return svm.SVC(
                max_iter=1000,
                probability=True
            )
        else:
            raise Exception(
                'Unsupported classifier. Available classifiers:' + 'multi-layer-perceptron, ' + 'linear-discriminant-analysis, ' + 'logistic-regression, ' + 'support-vector-machine')


configuration = Configuration()

dataset_path = configuration.get_dataset_path()
submission_path = configuration.get_submission_path()
current_id = str(uuid.uuid4())
subjects = range(configuration.get_subject_range_start(), configuration.get_subject_range_end())
ids_tot = []
pred_tot = []
error_tot = []
results = []
b, a = create_pre_filter(configuration)

for subject in subjects:
    epochs_tot = []
    train_files, test_files = cross_validation_prepare(8, subject, configuration.get_dataset_path())
    # read and concatenate all the files
    train_raw = concatenate_raws([configuration.load_data(train_file) for train_file in train_files])

    # pick eeg signal
    picks = pick_types(train_raw.info, eeg=True)

    # Filter data for alpha frequency and beta band
    train_raw._data[picks] = np.array(
        Parallel(n_jobs=configuration.get_maximum_paralel_jobs())(
            delayed(lfilter)(b, a, train_raw._data[i]) for i in picks))

    # Train feature extractor
    (feature_extractor, extractor_file) = train_feature_extractor(train_raw,
                                                                  picks,
                                                                  configuration)

    # Preprocess training data
    training_data = preprocess_data(train_raw, picks, configuration,
                                    trained_feature_extractor=feature_extractor)

    # training labels
    labels = np.asarray(train_raw._data[32:], dtype=np.float32)
    del train_raw

    # read test data
    test_data = [creat_mne_raw_object(test_file, read_events=False) for test_file in test_files]
    test_labels = [test_file.replace('_data', '_events') for test_file in test_files]
    test_raw = concatenate_raws(test_data)
    test_raw._data[picks] = np.array(
        Parallel(n_jobs=configuration.get_maximum_paralel_jobs())(
            delayed(lfilter)(b, a, test_raw._data[i]) for i in picks))

    # read ids
    ids = np.concatenate([np.array(pd.read_csv(test_file)['id']) for test_file in test_files])
    ids_tot.append(ids)

    # Preprocess test data
    test_data = preprocess_data(test_raw, picks, configuration, trained_feature_extractor=feature_extractor)
    del test_raw

    predictor = get_classifier(configuration)

    predictions = np.empty((len(ids), 6))
    events = configuration.get_events()
    print('Starting classifier training')
    for i in range(1, len(events)):
        if configuration.should_train_classifier():
            print('Training subject %d, class %s with %s predictor' % (
                subject, events[i], configuration.get_classifier_type()))
            predictor.fit(training_data[:, ::configuration.get_subsamples()].T,
                          labels[i, ::configuration.get_subsamples()])
            print('Done!')
        print('Testing...')
        predictions[:, i] = predictor.predict_proba(test_data.T)[:, 1]
        print('Done!')

    classifier_file = configuration.save_model(predictor, 'subject-%d' % subject)

    actual_events_proba = np.transpose(np.concatenate([extract_events_from_file(file)[0] for file in test_labels]))
    mean_squared_error = metrics.mean_absolute_error(actual_events_proba, predictions)
    roc_score = metrics.roc_auc_score(actual_events_proba, predictions)
    roc_auc_score = metrics.roc_auc_score(actual_events_proba, predictions)
    current_result = (
        configuration.get_classifier_type(),
        classifier_file,
        configuration.get_feature_extractor_type(),
        extractor_file,
        configuration.get_subsamples(),
        configuration.get_feature_extractor_settings()['number-of-filters'],
        configuration.get_feature_extractor_settings()['regularization'],
        configuration.get_pre_filtering_settings()['order'],
        configuration.get_smoothing_window_size(),
        configuration.get_smoothing_type(),
        subject,
        mean_squared_error,
        roc_score,
        roc_auc_score,
        0,
        0,
        time.strftime('%d/%m/%Y'),
        time.strftime('%H:%M:%S'),
        current_id
    )
    # pred_tot.append(predictions)
    print('Saving results...')
    configuration.save_result([current_result])
    print('Done!')

# print('Creating submission file')
# # create pandas object for submission
# submission = pd.DataFrame(index=np.concatenate(ids_tot),
#                           columns=events,
#                           data=np.concatenate(pred_tot))
#
# # write file
# submission.to_csv('%s%s_%s.csv' % (submission_path, configuration.get_classifier_type(), time.strftime('%Y%m%d%H%M%S')),
#                   index_label='id',
#                   float_format='%.5f')
