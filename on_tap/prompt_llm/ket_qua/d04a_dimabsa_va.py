import torch
import os, random, json, math, re, time, unicodedata, heapq
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.io import wavfile
from scipy.sparse import hstack
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, StratifiedGroupKFold,
                                     GroupShuffleSplit, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


seed_all(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def amp():
    return torch.autocast(dev, torch.bfloat16, enabled=use_bf16)
train_lines = [json.loads(l) for l in open("data/a_train.jsonl", encoding="utf-8")]
test_lines = [json.loads(l) for l in open("data/a_test.jsonl", encoding="utf-8")]

def clean(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"\b(ko|k|khong)\b", "không", s)
    s = re.sub(r"\b(dc|đc)\b", "được", s)
    return s

def clause(text, asp):
    t = clean(text)
    a = clean(asp)
    parts = re.split(r"[,.;!?]| nhưng | mà | còn | và ", t)
    for part in parts:
        if a in part:
            return part.replace(a, " ASP ")
    return t.replace(a, " ASP ")

train_rows = []
for line in train_lines:
    for asp in line["aspects"]:
        c = clause(line["text"], asp["aspect"])
        train_rows.append((c, asp["valence"], asp["arousal"]))

test_rows = []
for i, line in enumerate(test_lines):
    for asp in line["aspects"]:
        c = clause(line["text"], asp["term"])
        test_rows.append((i, asp["term"], c))

vectorizer_word = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=1)
vectorizer_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), sublinear_tf=True, min_df=2)

X_train_word = vectorizer_word.fit_transform([row[0] for row in train_rows])
X_train_char = vectorizer_char.fit_transform([row[0] for row in train_rows])
X_train = hstack([X_train_word, X_train_char])

y_train_v = [row[1] for row in train_rows]
y_train_a = [row[2] for row in train_rows]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_v = Ridge(alpha=0.1)
model_a = Ridge(alpha=0.1)

pv = cross_val_predict(model_v, X_train, y_train_v, cv=kf)
pa = cross_val_predict(model_a, X_train, y_train_a, cv=kf)

rmse_v = np.sqrt(np.mean((pv - y_train_v)**2))
rmse_a = np.sqrt(np.mean((pa - y_train_a)**2))
print(f"RMSE_VA: {np.sqrt((rmse_v**2 + rmse_a**2)/2)}")

model_v.fit(X_train, y_train_v)
model_a.fit(X_train, y_train_a)

X_test_word = vectorizer_word.transform([row[2] for row in test_rows])
X_test_char = vectorizer_char.transform([row[2] for row in test_rows])
X_test = hstack([X_test_word, X_test_char])

pred_v = model_v.predict(X_test)
pred_a = model_a.predict(X_test)

pred_v = np.clip(pred_v, 1, 9)
pred_a = np.clip(pred_a, 1, 9)

pred_v = np.round(pred_v, 2)
pred_a = np.round(pred_a, 2)

output = []
k=0
for line in test_lines:
    aspects = []
    for t in line["aspects"]:
        aspects.append({"aspect": t["term"], "valence": float(pred_v[k]), "arousal": float(pred_a[k])})
        k += 1
    output.append({"id": line["id"], "aspects": aspects})

with open("pred_a.jsonl", "w", encoding="utf-8") as f:
    for line in output:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
