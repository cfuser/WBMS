import csv
def read_file(name):
    f = open(name)
    csv_reader = csv.reader(f)
    feat = []
    label = []
    for line in csv_reader:
        feat.append(list(map(int, line[:-1])))
        label.append(list(map(int, line[-1])))
    return feat, label