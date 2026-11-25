"""Baseline dịch: không dịch, chỉ truy hồi câu tiếng Anh giống nhất trong train (TF-IDF) rồi trả về bản dịch của câu đó.
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
tr, te = pd.read_csv(D / "train.csv", keep_default_na=False), pd.read_csv(D / "test.csv", keep_default_na=False)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
A = vec.fit_transform(tr.en); B = vec.transform(te.en)
best = np.concatenate([np.asarray((B[i:i + 200] @ A.T).argmax(1)).ravel() for i in range(0, B.shape[0], 200)])
pd.DataFrame({"id": te.id, "vi": tr.vi.values[best]}).to_csv("submission.csv", index=False)
print("xong")
