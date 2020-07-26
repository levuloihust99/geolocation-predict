import pandas as pd
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json

def load_corpus():
    """return a corpus or a collection of documents, along with userID of each document. """

    data = pd.read_csv("dataset/textdata/filter_data.csv", header=0)
    uids = data.loc[:, "FacebookID"].values.astype(str)
    corpus = data.loc[:, "Text"].values.astype(str).tolist()

    # vietnamese tokenizing the corpus
    vn_corpus = []
    for doc in corpus:
        doc = ViTokenizer.tokenize(doc)
        # only keep 1000 first words of the document, the remaining words are ignored
        vn_corpus.append(doc[:1000])

    return uids, vn_corpus

def compute_tfidf_feature(uids, corpus):
    """
    Learn the tfidf feature from the supplied corpus.
    The result is stored in a csv file with each row consisting of a userID and the corresponding tfidf vector
    """
    # remove words that have document frequency higher than 0.5
    # and build the vocabulary with max size equal to 10000
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000)
    tfidfvecs = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.vocabulary_
    with open("vocabulary.json", "w") as fw:
        json.dump(vocabulary, fw)
    tfidfvecs = pd.DataFrame(tfidfvecs.toarray())
    uids = pd.Series(uids)
    compact = pd.concat([uids, tfidfvecs], axis=1)
    tfidfvecs_fname = "dataset/processed_data/tfidfvecs.csv"
    compact.to_csv(tfidfvecs_fname, index=False, header=False)

def compute_docvec_feature(uids, corpus):
    """
    Learn the vector representation of each document in the corpus using Doc2Vec model.
    The result is stored in a csv file with each row consisting of a userID and the corresponding document vector representation
    """

    model = Doc2Vec(vector_size=60, min_count=1, epochs=50)
    words_list = [doc.split(" ") for doc in corpus]
    tag_docs = []
    for i, words in enumerate(words_list):
        tag_docs.append(TaggedDocument(words=words, tags=[i]))

    print("Build vocabulary ....")
    model.build_vocab(tag_docs)
    print("Training ....")
    model.train(tag_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec.model")

    docvecs = [model.docvecs[i] for i in range(len(uids))]
    docvecs = pd.DataFrame(docvecs)
    uids = pd.Series(uids)
    docvecs = pd.concat([uids, docvecs], axis=1)

    docvecs_fname = "dataset/processed_data/docvecs.csv"
    docvecs.to_csv(docvecs_fname, index=False, header=False)

