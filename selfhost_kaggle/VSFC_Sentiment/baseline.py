"""Baseline cảm xúc: TF-IDF từ (1-2 gram) + LogisticRegression.
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
tr, te = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(tr.text), tr.label)
pd.DataFrame({"id": te.id, "label": clf.predict(vec.transform(te.text))}).to_csv("submission.csv", index=False)
print("xong")
