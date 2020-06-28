import pandas as pd
from model import getGeoPredictModel
import tensorflow as tf
import json
import numpy as np
import time
from datetime import datetime, date
import glob
import os
from utils import Split
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import re

def load_tfidfvecs(fname):
   tfidfvecs = pd.read_csv(fname, header=None).iloc[:, 1:].values.astype('float32')
   return tfidfvecs

def load_docvecs(fname):
    docvecs = pd.read_csv(fname, header=None).iloc[:, 1:].values.astype('float32')
    return docvecs

def str2opt(name, learning_rate):
    assert isinstance(learning_rate, float)

    if name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name.lower() == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif name.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)

if __name__ == "__main__":

    # run mode
    begin = int(round(time.time() * 1000))
    print("Parsing ...")

    parser = argparse.ArgumentParser(description="Provide appropriate parameters to choose running mode")
    parser.add_argument("--from-scratch", "-f", required=True, type=eval,
                        help="True if training model from scratch, False otherwise")
    parser.add_argument("--process-input", "-p", default=False, type=eval,
                        help="whether to perform transformation from raw text data to vector, default=True")
    parser.add_argument("--docvec-pretrain", "-d", default=False, type=eval,
                        help="whether to use pretrain doc2vec model, default=False, only use when process_input=True")
    parser.add_argument("--visualize", "-v", default=False, type=eval,
                        help="whether to visualize the label distribution before and after splitting the dataset, default=False")
    parser.add_argument("--ratio", "-t", default=0.2, type=eval,
                        help="fraction of test set in the dataset, default=0.2")
    parser.add_argument("--learning-rate", "-lr", default=0.1, type=eval,
                        help="learning rate of the optimizer, default=0.1")
    parser.add_argument("--optimizer", "-o", default='SGD',
                        help="optimizer to optimize the model, default='SGD'")
    parser.add_argument("--epochs", default=20, type=int,
                        help="number of epochs when training model")
    args = parser.parse_args()

    from_scratch = args.from_scratch
    process_input = args.process_input
    docvec_pretrain = args.docvec_pretrain
    visualize = args.visualize
    ratio=args.ratio
    learning_rate = args.learning_rate
    optimizer = str2opt(args.optimizer, learning_rate)
    epochs = args.epochs

    end = int(round(time.time() * 1000))
    print("Parse done! - Elapsed time: %d" % (end - begin))

    # process input

    if process_input:
        print("Processing input ...")
        begin = int(round(time.time() * 1000))
        import data_process

        docvec_pretrain = args.docvec_pretrain
        print("Loading text data ...")
        uids, corpus = data_process.load_corpus()
        print("Learning tfidf feature ...")
        data_process.compute_tfidf_feature(uids, corpus)
        if not docvec_pretrain:
            print("Learning Doc2Vec feature ...")
            data_process.compute_docvec_feature(uids, corpus)
        else:
            print("Load pretrained Doc2Vec model ...")
            docvec_model = data_process.Doc2Vec.load("doc2vec.model")
            docvecs = [docvec_model.docvecs[i] for i in range(len(uids))]
            docvecs = pd.DataFrame(docvecs)
            docvecs = pd.concat([uids, docvecs], axis=1)

            docvecs_fname = "dataset/processed_data/docvecs.csv"
            docvecs.to_csv(docvecs_fname, index=False, header=False)

        end = int(round(time.time() * 1000))
        print("Process input done! - Elapsed time: %d" % (end - begin))

    # load pre-processed data
    begin = int(round(time.time() * 1000))
    print("Loading pre-processed data ...")
    tfidfvecs_fname = "dataset/processed_data/tfidfvecs.csv"
    docvecs_fname = "dataset/processed_data/docvecs.csv"
    tfidfvecs = load_docvecs(tfidfvecs_fname)
    docvecs = load_docvecs(docvecs_fname)
    end = int(round(time.time() * 1000))
    print("Load data done! - Elapsed time: %d" % (end - begin))

    # compute labels
    begin = int(round(time.time() * 1000))
    filter_data_fname = "dataset/textdata/filter_data.csv"
    data_df = pd.read_csv(filter_data_fname, header=0)
    labels = []

    raw2true= {}
    raw2true_df = pd.read_csv("handtagged/place_mapping.csv", header=0)
    for i in range(raw2true_df.shape[0]):
        raw2true[raw2true_df.loc[i, "Raw_place"]] = raw2true_df.loc[i, "True_place"]
    with open("place2int.json", "r") as fp:
        place2int = json.load(fp)

    for i in range(data_df.shape[0]):
        label = raw2true[data_df.loc[i, "Place"]]
        labels.append(place2int[label])

    labels = np.array(labels)
    onehot_labels = np.zeros((labels.shape[0], 63), dtype='float32')
    onehot_labels[np.arange(labels.shape[0]), labels - 1] = 1
    end = int(round(time.time() * 1000))
    print("Compute labels done! - Elapsed time: %d" % (end - begin))

    # split train test
    begin = int(round(time.time() * 1000))
    split = Split(tfidfvecs=tfidfvecs, docvecs=docvecs, labels=onehot_labels, ratio=0.2)
    train_dataset, test_dataset = split.hold_out()
    end = int(round(time.time() * 1000))
    print("Split train test done! - Elapsed time: %d" % (end - begin))

    # visualize label distribution
    if visualize:
        begin = int(round(time.time() * 1000))

        # Draw label distribution before splitting
        bef_fig = plt.figure(figsize=(12, 5), num="Prior label distribution")
        chart = bef_fig.add_subplot()
        sns.distplot(labels, ax=chart)

        # Draw label distribution after splitting
        aft_fig = plt.figure(figsize=(12, 5), num="Posterior label distribution")
        chart = aft_fig.add_subplot()
        train_size = train_dataset["outputs"].shape[0]
        aft_labels = np.zeros(train_size)

        for i in range(train_size):
            idx = np.where(train_dataset["outputs"][i] == 1.0)[0][0] + 1
            aft_labels[i] = idx
        sns.distplot(aft_labels, ax=chart)

        plt.show()

    checkpoint_dir = "pretrained/"
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="{0}{1}-".format(checkpoint_dir, date.today()) + "-accuracy:{val_accuracy:.4f}.hdf5",
        save_weights_only=False,
        verbose=1,
        save_best_only=True,
        monitor='val_accuracy',
        save_freq='epoch',
        )

    from_scratch = args.from_scratch
    if from_scratch:
        tfidf_dim = tfidfvecs.shape[-1]
        docvec_dim = docvecs.shape[-1]
        model = getGeoPredictModel(tfidf_input_dim=tfidf_dim, doc2vec_input_dim=docvec_dim, tfidf_hidden_dim=150, doc2vec_hidden_dim=30, output_dim=63)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        model.fit(train_dataset['inputs'], train_dataset['outputs'], epochs=epochs,
                  batch_size=10, callbacks=[callback], validation_data=(test_dataset['inputs'], test_dataset['outputs']))
    else:
        list_files = glob.glob("./pretrained/*")
        # model_path = max(list_files, key=os.path.getctime)
        fnames_acc = []
        for file in list_files:
            fname_acc = re.findall(r"([\d\.]+)\.hdf5", file)
            fnames_acc.append(float(fname_acc[0]))
        fnames_acc.sort()
        max_acc = "{:.4f}".format(fnames_acc.pop())
        model_path = None
        for file in list_files:
            fname_acc = re.findall(r"([\d\.]+)\.hdf5", file)
            if fname_acc == max_acc:
                model_path = file

        assert model_path is not None

        model = tf.keras.models.load_model(model_path)
        model.fit(train_dataset['inputs'], train_dataset['outputs'], epochs=epochs,
                  batch_size=64, callbacks=[callback], validation_data=(test_dataset['inputs'], test_dataset['outputs']))

    end = int(round(time.time() * 1000))
    print("Train model done! - Elapsed time: %d" % (end - begin))