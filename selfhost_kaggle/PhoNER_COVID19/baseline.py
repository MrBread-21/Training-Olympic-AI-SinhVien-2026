"""Baseline NER: mỗi âm tiết lấy nhãn hay gặp nhất của nó trong train (không có thì O).
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
tr, te = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
dem = defaultdict(Counter)
for w, t in zip(tr.words, tr.tags):
    for a, b in zip(w.split(" "), t.split(" ")):
        dem[a][b] += 1
nhan = {w: c.most_common(1)[0][0] for w, c in dem.items()}
pd.DataFrame({"id": te.id, "tags": [" ".join(nhan.get(a, "O") for a in w.split(" ")) for w in te.words]}).to_csv("submission.csv", index=False)
print("xong")
