import csv
from glob import glob

from joblib import Parallel, delayed

from config.configuration import Configuration
from src.utils.utils import DTWDistance


def split_by_subject(file_name) -> list:
    file = open(file_name)
    csvf = csv.reader(file)
    next(csvf)
    subjects = [[], [], [], [], [], [], [], [], [], [], [], []]
    print('Reading CSV file %s' % file_name)
    for row in csvf:
        subject = int(float(row[6])) - 1
        values = list(map(float, row[0:-1]))
        if len(subjects[subject]) > 1225:
            continue
        subjects[subject].append(values)
    return subjects


def estimate_distances(sub1, subjects_1, subjects_2, config):
    results = []
    other_subs = list(range(len(subjects_1)))
    del other_subs[sub1]
    for sub2 in other_subs:
        print('Beginning DWT on %s subject %d and %s subject %d' % (move_sub_1, sub1 + 1, move_sub_2, sub2 + 1))
        distance = DTWDistance(subjects_1[sub1], subjects_2[sub2], 10)
        print('Distance between %s subject %d and %s subject %d = %f' % (
            move_sub_1, sub1 + 1, move_sub_2, sub2 + 1, distance))
        results.append([move_sub_1, move_sub_2, move_sub_1 == move_sub_2, sub1 + 1, sub2 + 1, distance])
    config.save_result(results, ['Movement1', 'Movement2', 'SameClass', 'Subject1', 'Subject2', 'Distance'])


files = glob('../extracted_data/*.csv')
configuration = Configuration()

for fname in files:
    for fname2 in files:
        subjects_1 = split_by_subject(fname)
        subjects_2 = split_by_subject(fname2)
        move_sub_1 = str(fname).replace('../extracted_data\\', '').replace('.csv', '')
        move_sub_2 = str(fname2).replace('../extracted_data\\', '').replace('.csv', '')
        Parallel(n_jobs=configuration.get_maximum_parallel_jobs())(
            delayed(estimate_distances)(sub1, subjects_1, subjects_2, configuration) for sub1 in range(len(subjects_1)))
