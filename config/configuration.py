import json
import os
import pathlib
import time
import csv
from os.path import exists

import joblib
from typing import Any, List


class Configuration:
    _config: dict

    # ----------------------------------------------------------------------------------------------------------------------#

    def __init__(self):
        configuration_file = open('../config/configuration.json', 'r')
        self._config = json.load(configuration_file)
        configuration_file.close()

    # ----------------------------------------------------------------------------------------------------------------------#

    # region General Settings

    def get_general_settings(self) -> dict:
        return self._config['general']

    def get_events(self) -> List[str]:
        return self.get_general_settings()['events']

    def get_maximum_paralel_jobs(self) -> int:
        return self.get_general_settings()['maximum-paralel-jobs']

    def get_subsamples(self) -> int:
        return self.get_general_settings()['subsamples']

    def get_cross_validation_folds(self) -> int:
        return self.get_general_settings()['cross-validation-folds']

    def get_dataset_path(self) -> str:
        return self.get_general_settings()['dataset-path']

    def get_trained_models_path(self) -> str:
        return self.get_general_settings()['trained-models-path']

    def get_trained_extractors_path(self) -> str:
        return self.get_general_settings()['trained-extractors-path']

    def get_result_path(self) -> str:
        return self.get_general_settings()['result-path']

    def _get_result_headers(self) -> List[str]:
        return ['classifier',
                'classifier_file',
                'extractor',
                'extractor_file',
                'subsampling',
                'nfilter',
                'regularization',
                'pre_filter_order',
                'smoothing_window',
                'smoothing_type',
                'subject',
                'error',
                'date',
                'time',
                'id']

    def save_result(self, result: List[Any]):
        should_create_headers = False
        result_path = self.get_result_path()
        if not exists(result_path):
            pathlib.Path.touch(pathlib.Path(result_path))
            should_create_headers = True
        file = open(self.get_result_path(), 'a', newline='\n', encoding='utf-8')
        writer = csv.writer(file)
        if should_create_headers:
            writer.writerow(self._get_result_headers())
        writer.writerows(result)
        file.close()

    def get_submission_path(self) -> str:
        return self.get_general_settings()['submission-path']

    def get_sampling_frequency(self) -> int:
        return self.get_general_settings()['sampling-frequency']

    def get_subject_range(self) -> dict:
        return self.get_general_settings()['subject-range']

    def get_subject_range_start(self) -> int:
        return self.get_subject_range()['start']

    def get_subject_range_end(self) -> int:
        return self.get_subject_range()['end']

    # end_region General Settings

    # ----------------------------------------------------------------------------------------------------------------------#

    # region Pre-filtering

    def get_pre_filtering_settings(self) -> dict:
        return self._config['pre-filtering']

    def get_pre_filtering_type(self) -> str:
        return self.get_pre_filtering_settings()['type']

    # end_region Pre-filtering

    # ----------------------------------------------------------------------------------------------------------------------#

    # region Feature extraction

    def get_feature_extractor_settings(self) -> dict:
        return self._config['feature-extractor']

    def should_train_extractor(self) -> bool:
        return self.get_feature_extractor_settings()['should-train'] is True

    def _should_save_extractor(self) -> bool:
        return self.get_feature_extractor_settings()['should-save'] is True

    def get_preload_extractor_file(self) -> str:
        return self.get_feature_extractor_settings()['preload-model-file']

    def _get_preload_extractor_path(self) -> str:
        return self.get_trained_extractors_path() + self.get_preload_extractor_file()

    def get_preloaded_extractor(self) -> Any:
        return joblib.load(self._get_preload_extractor_path())

    def has_preloaded_extractor(self):
        return self.get_feature_extractor_settings()['preload-model-file'] is not None

    def save_extractor(self, extractor: Any, prefix: str = "", sufix: str = "") -> str:
        if self._should_save_extractor():
            if self._should_save_classifier():
                filename = "%s%s_%s_%s_%s.sav" % (
                    self.get_trained_extractors_path(),
                    prefix,
                    self.get_feature_extractor_type(),
                    time.strftime('%Y%m%d%H%M%S'),
                    sufix
                )
                joblib.dump(extractor, filename)
                return filename
            return None

    def get_feature_extractor_type(self) -> str:
        return self.get_feature_extractor_settings()['type']

    def get_event_window_before(self) -> int:
        return self.get_feature_extractor_settings()['event-window-before']

    def get_event_window_after(self) -> int:
        return self.get_feature_extractor_settings()['event-window-after']

    def get_smoothing_window_size(self) -> int:
        return self.get_feature_extractor_settings()['smoothing-window-size']

    def get_smoothing_type(self) -> str:
        return self.get_feature_extractor_settings()['smoothing-type']

    # end_region Feature extraction

    # ----------------------------------------------------------------------------------------------------------------------#

    # region Classifier

    def get_classifier_settings(self) -> dict:
        return self._config['classifier']

    def get_classifier_type(self) -> str:
        return self.get_classifier_settings()['type']

    def _get_preload_model_path(self) -> str:
        return self.get_trained_models_path() + self.get_classifier_settings()['preload-model-file']

    def get_preloaded_model(self) -> Any:
        return joblib.load(self._get_preload_model_path())

    def has_preloaded_model(self):
        return self.get_classifier_settings()['preload-model-file'] is not None

    def save_model(self, classifier: Any, prefix: str = "", sufix: str = "") -> str:
        if self._should_save_classifier():
            filename = "%s%s_%s_%s_%s.sav" % (
                self.get_trained_models_path(),
                prefix,
                self.get_classifier_type(),
                time.strftime('%Y%m%d%H%M%S'),
                sufix)
            joblib.dump(classifier, filename)
            return filename
        return None

    def should_train_classifier(self):
        return self.get_classifier_settings()['should-train'] is True

    def _should_save_classifier(self):
        return self.get_classifier_settings()['should-save'] is True

    # end_region Classifier

# ----------------------------------------------------------------------------------------------------------------------#
