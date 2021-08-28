import os
import csv
import multiprocessing


def generate_new_csv_from_data(directory: str):
    for filename in os.listdir(directory):
        if 'data' in filename:
            print('Reading ' + filename + '...')
            join_data_event(directory + filename)
            print('Done!')
        else:
            continue


def join_data_event(filename):
    data_file = open(filename)
    events_file = open(filename.replace('_data', '_events'))
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\"')
    event_reader = csv.reader(events_file, delimiter=',', quotechar='\"')
    data_headers = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
                    'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
                    'O2', 'PO10']
    event_headers = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']
    next(data_reader, None)
    next(event_reader, None)
    new_data = []
    new_event = []
    i = 1
    while True:
        try:
            data_row = next(data_reader)
            event_row = next(event_reader)
        except StopIteration:
            break
        except Exception as e:
            print("Linha " + str(i) + " >> READ_ERROR >> ", e)
            i += 1
            continue
        del event_row[0]
        del data_row[0]
        i += 1
        new_data.append(data_row)
        new_event.append(event_row)
    treated_filename = filename.replace('train', 'treated')
    treated_data = open(treated_filename, 'w', newline='', encoding='utf-8')
    data_writer = csv.writer(treated_data)
    data_writer.writerow(data_headers)
    data_writer.writerows(new_data)
    treated_events = open(treated_filename.replace('_data', '_events'), 'w', newline='', encoding='utf-8')
    event_writer = csv.writer(treated_events)
    event_writer.writerow(event_headers)
    event_writer.writerows(new_event)


generate_new_csv_from_data('../../dataset/train/')
