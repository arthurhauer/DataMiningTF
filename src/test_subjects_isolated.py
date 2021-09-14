import time
import pandas as pd
import uuid
from mne import concatenate_raws, pick_types

from scipy.signal import lfilter

from sklearn import metrics

from src.utils.processing_utils.processing_utils import *
from src.utils.utils import cross_validation_prepare, creat_mne_raw_object, extract_events_from_file

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
        for iteration in range(0, 8):
            train_files, train_series, test_files, test_series = cross_validation_prepare(iteration, subject,
                                                                                          configuration.get_dataset_path())
            loaded_training_data, loaded_training_labels = configuration.load_extracted_data(subject, train_series)
            training_data = None
            labels = None
            extractor_file = None
            picks = None
            if loaded_training_data is None:
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
            else:
                training_data = loaded_training_data
                labels = loaded_training_labels
            del loaded_training_data
            del loaded_training_labels

            # read test data
            loaded_test_data, loaded_test_labels = configuration.load_extracted_data(subject, test_series)
            test_data = None
            test_labels = [test_file.replace('_data', '_events') for test_file in test_files]
            if loaded_test_data is None:
                test_data = [creat_mne_raw_object(test_file, read_events=False) for test_file in test_files]
                test_raw = concatenate_raws(test_data)
                picks = pick_types(test_raw.info, eeg=True)
                test_raw._data[picks] = np.array(
                    Parallel(n_jobs=configuration.get_maximum_parallel_jobs())(
                        delayed(lfilter)(b, a, test_raw._data[i]) for i in picks))
                test_data, _ = preprocess_data(test_raw, picks, configuration, subject, test_series,
                                               trained_feature_extractor=feature_extractor)
                del test_raw
            else:
                test_data = loaded_test_data
            # read ids
            ids = np.concatenate([np.array(pd.read_csv(test_file)['id']) for test_file in test_files])
            ids_tot.append(ids)

            # Preprocess test data
            del loaded_test_data
            del loaded_test_labels

            predictor = get_classifier(configuration)
            events = configuration.get_events()
            events_length = len(events)
            predictions = np.empty((len(ids), events_length))

            print('Starting classifier training')
            for i in range(1, events_length):
                if configuration.should_train_classifier():
                    print('Training subject %d, class %s with %s predictor' % (
                        subject, events[i], configuration.get_classifier_type()))
                    predictor.fit(training_data[:, ::configuration.get_subsamples()].T,
                                  labels[i, ::configuration.get_subsamples()])
                    print('Done!')
                print('Testing subject %d, class %s with %s predictor' % (
                    subject, events[i], configuration.get_classifier_type()))
                predictions[:, i] = predictor.predict_proba(test_data.T)[:, 1]
                print('Done!')

            classifier_file = configuration.save_model(predictor, 'subject-%d' % subject)

            actual_events_proba = np.transpose(
                np.concatenate([extract_events_from_file(file)[0] for file in test_labels]))
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
                current_id,
                test_series
            )
            print('Saving results...')
            configuration.save_result([current_result])
            print('Done!')
