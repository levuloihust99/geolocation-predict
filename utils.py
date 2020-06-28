import numpy as np

class Split(object):
    """
    Provide various functions to split the dataset into trainset and testset
    """
    def __init__(self, tfidfvecs, docvecs, labels, ratio=0.2):
        self._tfidfvecs = tfidfvecs
        self._docvecs = docvecs
        self._labels = labels
        self._ratio = ratio

    def hold_out(self):
        tfidfvecs = self._tfidfvecs
        docvecs = self._docvecs
        labels = self._labels
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        ratio = self._ratio

        batch_size = labels.shape[0]
        test_num = int(batch_size * ratio)
        indices = np.arange(batch_size)
        test_indices = np.random.choice(np.arange(batch_size), test_num, replace=False)
        test_dataset = {
            'inputs': [tfidfvecs[test_indices], docvecs[test_indices]],
            'outputs': labels[test_indices]
        }
        train_indices = np.array(list(set(indices) - set(test_indices)))
        train_dataset = {
            'inputs': [tfidfvecs[train_indices], docvecs[train_indices]],
            'outputs': labels[train_indices]
        }
        return train_dataset, test_dataset

    def stratified_sampling(self):
        pass