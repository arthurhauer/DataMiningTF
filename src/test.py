from typing import Any

import numpy as np
import pandas as pd
from mne.channels import make_standard_montage
from scipy.signal.windows import boxcar

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

from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


def creat_mne_raw_object(fname, read_events=True):
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
    return glob('../../dataset/train/subj%d_series[1-7]_data.csv' % subject_index), [
        '../../dataset/train/subj%d_series8_data.csv' % subject]


def train_feature_extractor(data, data_picks, extractor_type="csp", **kwargs) -> Any:
    during_tmin = kwargs.get('during_tmin')
    during_tmax = kwargs.get('during_tmax')
    before_tmin = kwargs.get('before_tmin')
    before_tmax = kwargs.get('before_tmax')
    if during_tmin is None:
        during_tmin = 0
    if during_tmax is None:
        during_tmax = 2
    if before_tmin is None:
        before_tmin = -2
    if before_tmax is None:
        before_tmax = 0
    y = []
    # get event position corresponding to HandStart
    events_data = find_events(data, stim_channel='HandStart', verbose=False)
    # epochs signal for 2 second after the event
    epochs = Epochs(data, events_data, {'during': 1}, during_tmin, during_tmax, proj=False,
                    picks=data_picks, baseline=None, preload=True,
                    verbose=False)

    epochs_tot.append(epochs)
    y.extend([1] * len(epochs))

    # epochs signal for 2 second before the event, this correspond to the
    # rest period.
    epochs_rest = Epochs(data, events_data, {'before': 1}, before_tmin, before_tmax, proj=False,
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
    if extractor_type == "csp":
        num_filters = kwargs.get('nfilters')
        regularization = kwargs.get('regularization')
        if num_filters is None:
            num_filters = 4
        if regularization is None:
            regularization = 'ledoit_wolf'
        # train CSP
        csp = CSP(n_components=num_filters, reg=regularization)
        csp.fit(x, y)
        return csp


def preprocess_data(data, data_picks, smoothing=None, trained_feature_extractor: Any = None,
                    job_count: int = -1) -> np.ndarray:
    extracted_data: Any
    processed_data: Any
    if trained_feature_extractor is None:
        extracted_data = data
    elif isinstance(feature_extractor, CSP):
        extracted_data = np.dot(trained_feature_extractor.filters_[0:nfilters], data._data[data_picks]) ** 2
    else:
        raise Exception("Unsupported feature extractor")
    if smoothing is not None:
        processed_data = np.array(
            Parallel(n_jobs=job_count)(
                delayed(convolve)(extracted_data[i], smoothing, 'full') for i in range(nfilters)))
    else:
        processed_data = extracted_data
    processed_data = np.log(processed_data[:, 0:extracted_data.shape[1]])
    processed_data = np.asarray(processed_data, dtype=np.float32)
    return processed_data


def get_classifier(chosen_classifier) -> Any:
    if chosen_classifier == "nn":
        return NeuralNet(
            layers=[  # three layers: one hidden layer
                ('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
            ],
            # layer parameters:
            input_shape=(None, 4),  # 96x96 input pixels per batch
            hidden_num_units=100,  # number of units in hidden layer
            output_nonlinearity=nonlinearities.sigmoid,  # output layer uses identity function
            output_num_units=1,  # 30 target values

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.001,
            update_momentum=0.9,

            regression=True,  # flag to indicate we're dealing with regression problem
            max_epochs=15,  # we want to train this many epochs
            # verbose=1,
        )
    elif chosen_classifier == 'lda':
        return LinearDiscriminantAnalysis()
    elif chosen_classifier == 'lr':
        return LogisticRegression()
    else:
        raise Exception('Unsupported classifier')


def extract_events_from_file(file) -> Any:
    events_raw = pd.read_csv(file)
    events_names = events_raw.columns[1:]
    events_data = np.array(events_raw[events_names]).T
    return events_data, events_names


def compare_predicted_and_actual(events_data, predicted):
    predicted_event_quadratic_error = np.square(np.subtract(events_data, predicted))
    return predicted_event_quadratic_error


subjects = range(1, 2)
ids_tot = []
pred_tot = []

# design a butterworth bandpass filter
freqs = [7, 30]
b, a = butter(5, np.array(freqs) / 250.0, btype='bandpass')

# CSP parameters
# Number of spatial filter to use
nfilters = 4

# convolution
# window for smoothing features
nwin = 250

# training subsample
subsample = 100

# max parallel jobs
max_job_count = 5

# submission file
submission_file = 'beat_the_benchmark.csv'

cols = ['HandStart', 'FirstDigitTouch',
        'BothStartLoadPhase', 'LiftOff',
        'Replace', 'BothReleased']

classifier = 'lda'

cross_validation_folds = 8

for subject in subjects:
    epochs_tot = []
    train_files, test_files = cross_validation_prepare(8, subject)
    # read and concatenate all the files
    train_raw = concatenate_raws([creat_mne_raw_object(train_file) for train_file in train_files])

    # pick eeg signal
    picks = pick_types(train_raw.info, eeg=True)

    # Filter data for alpha frequency and beta band
    train_raw._data[picks] = np.array(
        Parallel(n_jobs=max_job_count)(delayed(lfilter)(b, a, train_raw._data[i]) for i in picks))

    # Train feature extractor
    feature_extractor = train_feature_extractor(train_raw,
                                                picks,
                                                extractor_type="csp",
                                                num_filters=nfilters,
                                                regularization='ledoit_wolf',
                                                during_tmin=0,
                                                during_tmax=2,
                                                before_tmin=-2,
                                                before_tmax=0)

    # Preprocess training data
    training_data = preprocess_data(train_raw, picks, smoothing=boxcar(nwin),
                                    trained_feature_extractor=feature_extractor,
                                    job_count=max_job_count)

    # training labels
    labels = np.asarray(train_raw._data[32:], dtype=np.float32)
    del train_raw

    # read test data
    test_data = [creat_mne_raw_object(test_file, read_events=False) for test_file in test_files]
    test_labels = [test_file.replace('_data', '_events') for test_file in test_files]
    test_raw = concatenate_raws(test_data)
    test_raw._data[picks] = np.array(
        Parallel(n_jobs=max_job_count)(delayed(lfilter)(b, a, test_raw._data[i]) for i in picks))

    # read ids
    ids = np.concatenate([np.array(pd.read_csv(test_file)['id']) for test_file in test_files])
    ids_tot.append(ids)

    # Preprocess test data
    test_data = preprocess_data(test_raw, picks, smoothing=boxcar(nwin), trained_feature_extractor=feature_extractor,
                                job_count=max_job_count)
    del test_raw

    predictor = get_classifier(classifier)

    predictions = np.empty((len(ids), 6))
    for i in range(6):
        print('Training subject %d, class %s with %s predictor' % (subject, cols[i], classifier))
        predictor.fit(training_data[:, ::subsample].T, labels[i, ::subsample])
        print('Done! Testing...')
        predictions[:, i] = predictor.predict_proba(test_data.T)[:, 1]
        print('Done! Testing with cross validation')
        score = cross_val_score(predictor, training_data[:, ::subsample].T, labels[i, ::subsample])
        print("Done! Got the following scores: " + str(score))

    events = np.transpose(np.concatenate([extract_events_from_file(file)[0] for file in test_labels]))
    error = compare_predicted_and_actual(events, predictions)

    pred_tot.append(error)

# create pandas object for submission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file, index_label='id', float_format='%.5f')
