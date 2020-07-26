import re
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt

# Get data
prefix = 'https://www.facebook.com/'
identity = input("Enter a user's facebook link or user's facebook ID: ")
identity = re.sub(r"{}".format(prefix), "", identity)
identity = identity.strip()

headers = {'apikey': '9X3zZVEWmNKHWxncwA3PHrwbReNZZsrq'}
list_status_url = "http://124.158.1.123:8002/api/list/status"
list_status_payload = {
    'skip': 0,
    'limit': 30,
    'order_date': 'desc',
    'owner_type': 0,
}

if re.match(r"[a-z]", identity):
    profile_url = "http://124.158.1.123:8002/api/person/profile"
    res = requests.get(profile_url, headers=headers, params={'link': '/' + identity})
    identity = res.json()["data"]["fb_id"]
list_status_payload["owner_id"] = identity

list_status_response = requests.get(list_status_url, headers=headers, params=list_status_payload)
list_status_data = list_status_response.json()["data"]
document = [x["content"] for x in list_status_data]
# concatenate list of status into a single string, seperated by |
document = "|".join(document)

# preprocessing text
# remove links
document = re.sub(r"https?:\/\/[^\s]*", "", document)
document = re.sub(r"www[^\s]*", "", document)
# remove punctuation and non-vietnamese symbols
vn_accents = u'àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìí' \
             u'ĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữự' \
             u'ỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾ' \
             u'ỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤ' \
             u'ỦƯỨỪỬỮỰỲỴỶỸÝ'
regex = re.compile("[^{0}a-zA-Z0-9 ]".format(vn_accents))
document = regex.sub(" ", document)
# remove duplicated spaces
document = re.sub(' +', ' ', document)
# remove leading and trailing spaces
document = document.strip()
# lowering
document = document.lower()

with open("vocabulary.json", "r") as fr:
    vocab = json.load(fr)

vectorizer = TfidfVectorizer(vocabulary=vocab, max_features=10000)
tfidfvec = vectorizer.fit_transform([document])
docvec_model = Doc2Vec.load("doc2vec.model")
docvec = docvec_model.infer_vector(document.split())

tfidfvec = tfidfvec.toarray()
tfidfvec.shape = (1, 10000)
docvec.shape = (1, 60)

prefix = "./pretrained/multiview/"
list_files = glob.glob("{}*".format(prefix))
# model_path = max(list_files, key=os.path.getctime)
fnames_acc = []
for file in list_files:
    fname_acc = re.findall(r"([\d\.]+)\.hdf5", file)
    fnames_acc.append(float(fname_acc[0]))
fnames_acc.sort()
max_acc = "{:.4f}".format(fnames_acc[-1])
model_path = None
for file in list_files:
    fname_acc = re.findall(r"([\d\.]+)\.hdf5", file)
    fname_acc = float(fname_acc[0])
    if fname_acc == float(max_acc):
        model_path = file

assert model_path is not None

model = tf.keras.models.load_model(model_path)
pred = model.predict([tfidfvec, docvec])
pred.shape = 63

idx_sort = np.argsort(pred)
idxs = idx_sort[-5:] + 1

with open("int2place.json", "r") as fr:
    mapper = json.load(fr)

out = {}
for idx in idxs:
    out[mapper[str(idx)]] = pred[idx - 1]

x_labels = []
y_labels = []
for item in out.items():
    x_labels.append(item[0])
    y_labels.append(item[1])
plt.bar(x_labels, y_labels)
plt.show()