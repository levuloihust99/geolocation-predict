import pandas as pd
from model import getGeoPredictModel, getTextBasedModel
import tensorflow as tf
import json
import numpy as np
import time
from datetime import datetime, date
import glob
import os
import utils
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import re
import pickle

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
    parser.add_argument("--textbased-only", "-tx", default=False, type=eval,
                        help="whether to use text-based only model or multi-view model")
    parser.add_argument("--from-scratch", "-f", required=True, type=eval,
                        help="True if training model from scratch, False otherwise")
    parser.add_argument("--process-input", "-p", default=False, type=eval,
                        help="whether to perform transformation from raw text data to vector, default=True")
    parser.add_argument("--docvec-pretrain", "-d", default=False, type=eval,
                        help="whether to use pretrain doc2vec model, default=False, only use when process_input=True")
    parser.add_argument("--learning-rate", "-lr", default=0.01, type=eval,
                        help="learning rate of the optimizer, default=0.1")
    parser.add_argument("--optimizer", "-o", default='SGD',
                        help="optimizer to optimize the model, default='SGD'")
    parser.add_argument("--epochs", default=20, type=int,
                        help="number of epochs when training model")
    args = parser.parse_args()

    textbased_only = args.textbased_only
    from_scratch = args.from_scratch
    process_input = args.process_input
    docvec_pretrain = args.docvec_pretrain
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
        if not textbased_only:
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
    train_tfidfvecs = utils.load_vecs("dataset/processed_data/train/tfidfvecs.csv")
    test_tfidfvecs = utils.load_vecs("dataset/processed_data/test/tfidfvecs.csv")
    train_docvecs = utils.load_vecs("dataset/processed_data/train/docvecs.csv")
    test_docvecs = utils.load_vecs("dataset/processed_data/test/docvecs.csv")
    train_labels = utils.compute_labels("dataset/textdata/train/data.csv")
    test_labels = utils.compute_labels("dataset/textdata/test/data.csv")
    train_dataset = {
        'inputs': [train_tfidfvecs, train_docvecs],
        'outputs': [train_labels]
    }
    test_dataset = {
        'inputs': [test_tfidfvecs, test_docvecs],
        'outputs': [test_labels]
    }
    end = int(round(time.time() * 1000))
    print("Compute labels done! - Elapsed time: %d" % (end - begin))

    if not textbased_only:
        checkpoint_dir = "pretrained/multiview/"
    else:
        checkpoint_dir = "pretrained/textbased/"

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
        if not textbased_only:
            tfidf_dim = train_tfidfvecs.shape[-1]
            docvec_dim = train_docvecs.shape[-1]
            model = getGeoPredictModel(tfidf_input_dim=tfidf_dim, doc2vec_input_dim=docvec_dim, tfidf_hidden_dim=150, doc2vec_hidden_dim=30, output_dim=63)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy'])
            model.fit(train_dataset['inputs'], *train_dataset['outputs'], epochs=epochs,
                      batch_size=10, callbacks=[callback], validation_data=(test_dataset['inputs'], *test_dataset['outputs']))
        else:
            docvec_dim = train_docvecs.shape[-1]
            model = getTextBasedModel(doc2vec_input_dim=docvec_dim, doc2vec_hidden_dim=30, output_dim=63)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy'])
            model.fit(*train_dataset['inputs'], *train_dataset['outputs'], epochs=epochs,
                      batch_size=10, callbacks=[callback], validation_data=(*test_dataset['inputs'], *test_dataset['outputs']))
    else:
        if not textbased_only:
            prefix = "./pretrained/multiview/"
        else:
            prefix = "./pretrained/textbased/"
        list_files = glob.glob("{}*".format(prefix))

        fnames_acc = []
        for file in list_files:
            fname_acc = re.findall(r"([\d\.]+)\.hdf5", file)
            fnames_acc.append(float(fname_acc[0]))
        fnames_acc.sort()
        max_acc = "{:.4f}".format(fnames_acc.pop())
        model_path = None
        for file in list_files:
            fname_acc = re.findall(r"([\d\.]+)\.hdf5", file)
            fname_acc = float(fname_acc[0])
            if fname_acc == float(max_acc):
                model_path = file

        assert model_path is not None

        model = tf.keras.models.load_model(model_path)
        model.fit(train_dataset['inputs'], *train_dataset['outputs'], epochs=epochs,
                  batch_size=10, callbacks=[callback], validation_data=(test_dataset['inputs'], *test_dataset['outputs']))

    end = int(round(time.time() * 1000))
    print("Train model done! - Elapsed time: %d" % (end - begin))
