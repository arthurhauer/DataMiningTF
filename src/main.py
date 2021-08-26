import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import mne
tmin, tmax = -1., 4.
event_id = dict(still='0', movement='1')
# Read the CSV file as a NumPy array
data = np.transpose(np.loadtxt('../dataset/treated/subj1_series1.csv', delimiter=','))

# Some information about the channels
ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10',
            'Movement']

# Sampling rate of the Nautilus machine
sfreq = 500  # Hz

# Create the info structure needed by MNE
info = mne.create_info(ch_names, sfreq)

# Finally, create the Raw object
raw = mne.io.RawArray(data, info)

# # Plot it!
raw.plot()
events = mne.read_events('../dataset/train/subj1_series1_events.csv')
# eegbci.standardize(raw)  # set channel names
# montage = make_standard_montage('standard_1005')
# raw.set_montage(montage)
# # strip channel names of "." characters
# raw.rename_channels(lambda x: x.strip('.'))

# Apply band-pass filter
# raw.filter(7., 30., fir_design='firwin')
print('got here')
# events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
#                    exclude='bads')

# # Read epochs (train will be done only between 1 and 2s)
# # Testing will be done with a running classifier
# epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
#                 baseline=None, preload=True)
# epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
# labels = epochs.events[:, -1] - 2
# # Define a monte-carlo cross-validation generator (reduce variance):
# scores = []
# epochs_data = epochs.get_data()
# epochs_data_train = epochs_train.get_data()
# cv = ShuffleSplit(10, test_size=0.2, random_state=42)
# cv_split = cv.split(epochs_data_train)
#
# # Assemble a classifier
# lda = LinearDiscriminantAnalysis()
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
#
# # Use scikit-learn Pipeline with cross_val_score function
# clf = Pipeline([('CSP', csp), ('LDA', lda)])
# scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
#
# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                           class_balance))
#
# # plot CSP patterns estimated on full data for visualization
# csp.fit_transform(epochs_data, labels)
#
# csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)