import random, re, unicodedata
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
seed_all(42)

def chuan_hoa(s):
    s = unicodedata.normalize("NFC", str(s)).lower()                  # NFD và NFC thành cùng một chuỗi
    return re.sub(r"\s+", " ", s).strip()

tr, te = pd.read_csv("data/a_train.csv"), pd.read_csv("data/a_test.csv")
xtr, xte = tr.text.map(chuan_hoa), te.text.map(chuan_hoa)
word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)   # bền với lỗi gõ, mất dấu
Xtr = hstack([word.fit_transform(xtr), char.fit_transform(xtr)]).tocsr()
Xte = hstack([word.transform(xte), char.transform(xte)]).tocsr()
clf = LogisticRegression(C=5, max_iter=3000)
oof = cross_val_predict(clf, Xtr, tr.label, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print("CV macro-F1:", round(f1_score(tr.label, oof, average="macro"), 4))
print(confusion_matrix(tr.label, oof))
clf.fit(Xtr, tr.label)                                                 # bản cuối: toàn bộ train
pd.DataFrame({"id": te.id, "label": clf.predict(Xte)}).to_csv("submission_a.csv", index=False)
print("đã ghi submission_a.csv")
