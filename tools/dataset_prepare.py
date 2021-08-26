import os
import csv
import multiprocessing


def generate_new_csv_from_data(directory: str):
    pool = multiprocessing.Pool(4)
    for filename in os.listdir(directory):
        if 'data' in filename:
            print('Reading '+filename+'...')
            join_data_event(directory + filename)
            print('Done!')
        else:
            continue

def join_data_event(filename):
    data_file = open(filename)
    events_file = open(filename.replace('_data', '_events'))
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\"')
    event_reader = csv.reader(events_file, delimiter=',', quotechar='\"')
    headers = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
               'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10',
               # 'HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased',
               'Movement']
    next(data_reader, None)
    next(event_reader, None)
    new_data = []
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
        movement = '1' if any(element == '1' for element in event_row) else '0'
        data_row.append(movement)
        i += 1
        new_data.append(data_row)
    joined_data = open(filename.replace('_data', '').replace('train', 'treated'), 'w', newline='', encoding='utf-8')
    writer = csv.writer(joined_data)
    writer.writerow(headers)
    writer.writerows(new_data)


generate_new_csv_from_data('../dataset/train/')
