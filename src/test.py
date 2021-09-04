import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tsfel
from mne.channels import make_standard_montage
from scipy.signal.windows import boxcar
from sklearn import svm, neural_network

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from mne.io import RawArray
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from scipy.signal import butter, lfilter, convolve

from sklearn.linear_model import LogisticRegression
from glob import glob

from joblib import Parallel, delayed

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


def cross_validation_prepare(folds, subject_index) -> Any:
    if folds > 8 or folds < 2:
        raise Exception('Unsupported folds')
    return glob(dataset_path + 'subj%d_series[1-7]_data.csv' % subject_index), [
        dataset_path + 'subj%d_series8_data.csv' % subject]


def train_feature_extractor(data, data_picks, configuration_data: Configuration) -> Any:
    print('Starting feature extractor training')
    event_window_before = configuration_data.get_event_window_before()
    event_window_after = configuration_data.get_event_window_after()
    y = []
    # get event position corresponding to HandStart
    events_data = find_events(data, stim_channel='HandStart', verbose=False)
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
        csp = CSP(n_components=num_filters, reg=regularization)
        csp.fit(x, y)
        return csp

    elif extractor_type == 'tsfel':
        print('Training TSFEL')
        return tsfel.get_features_by_domain()
    else:
        raise Exception('Unsupported feature extractor. Available types: ' + 'csp, ' + 'tsfel')


def preprocess_data(data, data_picks, configuration_data: Configuration,
                    trained_feature_extractor: Any = None) -> np.ndarray:
    print('Starting data pre-process...')
    extracted_data: Any
    processed_data: Any
    number_of_filters = configuration_data.get_feature_extractor_settings()['number-of-filters']
    smoothing_type = configuration_data.get_smoothing_type()
    if smoothing_type is None:
        smoothing = None
    elif smoothing_type == 'boxcar':
        smoothing = boxcar(configuration_data.get_smoothing_window_size())
    else:
        raise Exception('Unsupported smoothing type. Available types: ' + 'boxcar')

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
            )
        elif chosen_classifier == 'linear-discriminant-analysis':
            return LinearDiscriminantAnalysis(

            )
        elif chosen_classifier == 'logistic-regression':
            return LogisticRegression(

            )
        elif chosen_classifier == 'support-vector-machine':
            return svm.SVC(
                max_iter=200,
                probability=True
            )
        else:
            raise Exception(
                'Unsupported classifier. Available classifiers:' + 'multi-layer-perceptron, ' + 'linear-discriminant-analysis, ' + 'logistic-regression, ' + 'support-vector-machine')


def extract_events_from_file(file) -> Any:
    events_raw = pd.read_csv(file)
    events_names = events_raw.columns[1:]
    events_data = np.array(events_raw[events_names]).T
    return events_data, events_names


def compare_predicted_and_actual(events_data, predicted):
    predicted_event_quadratic_error = np.absolute(events_data-predicted)
    return predicted_event_quadratic_error


configuration = Configuration()

dataset_path = configuration.get_dataset_path()
submission_path = configuration.get_submission_path()

subjects = range(configuration.get_subject_range_start(), configuration.get_subject_range_end())
ids_tot = []
pred_tot = []
error_tot=[]
b, a = create_pre_filter(configuration)

cols = ['HandStart', 'FirstDigitTouch',
        'BothStartLoadPhase', 'LiftOff',
        'Replace', 'BothReleased']

for subject in subjects:
    epochs_tot = []
    train_files, test_files = cross_validation_prepare(8, subject)
    # read and concatenate all the files
    train_raw = concatenate_raws([creat_mne_raw_object(train_file) for train_file in train_files])

    # pick eeg signal
    picks = pick_types(train_raw.info, eeg=True)

    # Filter data for alpha frequency and beta band
    train_raw._data[picks] = np.array(
        Parallel(n_jobs=configuration.get_maximum_paralel_jobs())(
            delayed(lfilter)(b, a, train_raw._data[i]) for i in picks))

    # Train feature extractor
    feature_extractor = train_feature_extractor(train_raw,
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
    print('Starting classifier training')
    for i in range(6):
        if configuration.should_train_classifier():
            print('Training subject %d, class %s with %s predictor' % (
                subject, cols[i], configuration.get_classifier_type()))
            predictor.fit(training_data[:, ::configuration.get_subsamples()].T,
                          labels[i, ::configuration.get_subsamples()])
            print('Done!')
        print('Testing...')
        predictions[:, i] = predictor.predict_proba(test_data.T)[:, 1]
        print('Done!')
        # score = cross_val_score(predictor, training_data[:, ::configuration.get_subsamples()].T,
        #                         labels[i, ::configuration.get_subsamples()])
        # print("Done! Got the following scores: " + str(score))

    configuration.save_model(predictor, 'subject-%d' % subject)

    events = np.transpose(np.concatenate([extract_events_from_file(file)[0] for file in test_labels]))
    error = compare_predicted_and_actual(events, predictions)
    print('Sum of error: '+str(np.sum(error)/error.shape[0]))
    pred_tot.append(predictions)

    print('Creating submission file')
# create pandas object for submission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv('%s%s_%s.csv' % (submission_path, configuration.get_classifier_type(), time.strftime('%Y%m%d%H%M%S')),
                  index_label='id',
                  float_format='%.5f')
