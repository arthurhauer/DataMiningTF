import json
import time

import joblib
from typing import Any


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

    def save_model(self, classifier: Any, prefix: str = "", sufix: str = ""):
        if self._should_save_classifier():
            joblib.dump(classifier, "%s%s_%s_%s_%s.sav" % (
                self.get_trained_models_path(), prefix, self.get_classifier_type(), time.strftime('%Y%m%d%H%M%S'),
                sufix))

    def should_train_classifier(self):
        return self.get_classifier_settings()['should-train'] is True

    def _should_save_classifier(self):
        return self.get_classifier_settings()['should-save'] is True

    # end_region Classifier

# ----------------------------------------------------------------------------------------------------------------------#
