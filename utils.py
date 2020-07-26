import numpy as np
import pandas as pd
import json
import pickle

def load_vecs(fname):
    vecs = pd.read_csv(fname, header=None).iloc[:, 1:].values.astype('float32')
    return vecs

def compute_labels(path):
    datadf = pd.read_csv(path, header=0)
    placedf = pd.read_csv("handtagged/place_mapping.csv", header=0)
    mapper = {}
    for i in range(placedf.shape[0]):
        mapper[placedf.loc[i, "Raw_place"]] = placedf.loc[i, "True_place"]
    with open("place2int.json", "r") as fr:
        place2int = json.load(fr)
    labels = []
    for i in range(datadf.shape[0]):
        raw_place = datadf.loc[i, "Place"]
        true_place = mapper[raw_place]
        labels.append(place2int[true_place])
    labels = np.array(labels)

    onehot_labels = np.zeros((labels.shape[0], 63), dtype='float32')
    onehot_labels[np.arange(labels.shape[0]), labels - 1] = 1
    return onehot_labels

class Split(object):
    def __init__(self, path="dataset/textdata/"):
        self.path = path

    def hold_out(self, ratio=0.2):
        datadf = pd.read_csv(self.path + "filter_data.csv", header=0)
        batch_size = datadf.shape[0]
        idxs = np.arange(batch_size)
        test_size = int(batch_size * ratio)
        test_idxs = np.random.choice(idxs, test_size, replace=False)
        train_idxs = np.array(list(set(idxs) - set(test_idxs)))
        train_set = datadf.iloc[train_idxs, :]
        train_set.to_csv(self.path + "train/data.csv", index=False)
        test_set = datadf.iloc[test_idxs, :]
        test_set.to_csv(self.path + "test/data.csv", index=False)

        storage = {"train_idxs": train_idxs, "test_idxs": test_idxs}
        with open("dataset/processed_data/idxs", "wb") as fw:
            pickle.dump(storage, fw)

        tfidfvecs = pd.read_csv("dataset/processed_data/tfidfvecs.csv", header=None)
        tfidfvecs_train = tfidfvecs.iloc[train_idxs, :]
        tfidfvecs_train.to_csv("dataset/processed_data/train/tfidfvecs.csv", header=False, index=False)
        tfidfvecs_test = tfidfvecs.iloc[test_idxs, :]
        tfidfvecs_test.to_csv("dataset/processed_data/test/tfidfvecs.csv", header=False, index=False)

        docvecs = pd.read_csv("dataset/processed_data/docvecs.csv", header=None)
        docvecs_train = docvecs.iloc[train_idxs, :]
        docvecs_train.to_csv("dataset/processed_data/train/docvecs.csv", header=False, index=False)
        docvecs_test = docvecs.iloc[test_idxs, :]
        docvecs_test.to_csv("dataset/processed_data/test/docvecs.csv", header=False, index=False)