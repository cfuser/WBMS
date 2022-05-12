import csv
from email import header
import numpy as np
def read_file(name):
    f = open(name)
    csv_reader = csv.reader(f)
    if 'GCM.csv' in name:
        csv_reader.__next__()
    feat = []
    label = []
    for line in csv_reader:
        # print(line)
        # print(list(line.strip()))
        # exit()
        feat.append(list(map(float, line[:-1])))
        # label.append(list(map(float, [line[-1]])))
        label.append(line[-1])
        # print(list(map(float, [line[-1]])))
        # exit()
        # feat.append(list(np.float_(line[:-1])))
        # label.append(list(np.float_(line[-1])))
    label_list = list(set(label))
    # print(list(set(label)))
    # print(list(range(len(label_list))))
    d = dict(zip(label_list, list(range(len(label_list)))))
    # print(d)
    res_label = [d[single_label] for single_label in label]
    # print(res_label)
    # exit()
    return feat, res_label