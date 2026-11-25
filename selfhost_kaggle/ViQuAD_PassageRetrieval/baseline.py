"""Baseline truy xuất: TF-IDF từ trên đoạn văn, lấy 10 đoạn có cosine cao nhất với câu hỏi.
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
cp, te = pd.read_csv(D / "corpus.csv"), pd.read_csv(D / "test.csv")
vec = TfidfVectorizer()
A = vec.fit_transform(cp.title + " " + cp.text); B = vec.transform(te.question)
S = (B @ A.T).toarray()
top = np.argsort(-S, axis=1)[:, :10]
pd.DataFrame({"id": te.id, "passage_ids": [" ".join(cp.passage_id.values[t]) for t in top]}).to_csv("submission.csv", index=False)
print("xong")
