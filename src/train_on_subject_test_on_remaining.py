import time
from glob import glob

import pandas as pd
import uuid
from mne import concatenate_raws, pick_types

from scipy.signal import lfilter

from sklearn import metrics

from src.utils.processing_utils.processing_utils import *
from src.utils.utils import cross_validation_prepare, creat_mne_raw_object, extract_events_from_file


def test_on_subject(config_data, test_subject, classifier):
    test_files = glob(dataset_path + 'subj%d_series[1-8]_data.csv' % test_subject)
    # read test data
    test_labels = [test_file.replace('_data', '_events') for test_file in test_files]
    test_data = [creat_mne_raw_object(test_file, read_events=False) for test_file in test_files]
    test_raw = concatenate_raws(test_data)
    picks = pick_types(test_raw.info, eeg=True)
    test_raw._data[picks] = np.array(
        Parallel(n_jobs=configuration.get_maximum_parallel_jobs())(
            delayed(lfilter)(b, a, test_raw._data[i]) for i in picks))
    test_data, _ = preprocess_data(test_raw, picks, configuration, subject, train_series,
                                   trained_feature_extractor=feature_extractor)
    del test_raw
    # read ids
    ids = np.concatenate([np.array(pd.read_csv(test_file)['id']) for test_file in test_files])
    ids_tot.append(ids)

    predictions = np.empty((len(ids), events_length))

    print('Starting test')
    for i in range(1, events_length):
        print('Testing subject %d, class %s with %s predictor' % (
            test_subject, events[i], config_data.get_classifier_type()))
        predictions[:, i] = classifier.predict_proba(test_data.T)[:, 1]
        print('Done!')

    classifier_file = config_data.save_model(classifier, 'subject-%d' % subject)

    actual_events_proba = np.concatenate(
        [np.transpose(np.asarray(extract_events_from_file(file)[0], dtype=np.float32)) for file in test_labels])

    roc_auc_score = metrics.roc_auc_score(actual_events_proba, predictions)
    current_result = (
        config_data.get_classifier_type(),
        config_data.get_feature_extractor_type(),
        config_data.get_event_window(),
        config_data.get_smoothing_window_size(),
        config_data.get_smoothing_type(),
        test_subject,
        roc_auc_score,
    )
    print('Saving results...')
    result_headers = ['classifier',
                      'extractor',
                      'window',
                      'smoothing_window',
                      'smoothing_type',
                      'test_subject',
                      'roc_score']
    config_data.save_result([current_result], result_headers)
    print('Done!')


configuration = Configuration()
for config_index in range(0, configuration.get_config_size()):
    configuration.set_index(config_index)
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
        train_files = glob(dataset_path + 'subj%d_series[1-8]_data.csv' % subject)
        train_series = list(range(1, 9))
        # read and concatenate all the files
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
        training_data, labels = preprocess_data(train_raw, picks, configuration, subject, train_series,
                                                trained_feature_extractor=feature_extractor)
        del train_raw

        predictor, grid = get_classifier(configuration)
        events = configuration.get_events()
        events_length = len(events)

        print('Starting classifier training')
        for i in range(1, events_length):
            print('Training subject %d, class %s with %s predictor' % (
                subject, events[i], configuration.get_classifier_type()))
            predictor.fit(training_data[:, ::configuration.get_subsamples()].T,
                          labels[i, ::configuration.get_subsamples()])
            print('Done!')

        classifier_file = configuration.save_model(predictor, 'subject-%d' % subject)

        test_subjects = list(range(1, 13))
        del test_subjects[subject - 1]
        Parallel(n_jobs=configuration.get_maximum_parallel_jobs())(
            delayed(test_on_subject)(configuration, test_subject, predictor) for test_subject in test_subjects)
