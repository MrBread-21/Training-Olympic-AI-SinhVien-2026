"""Baseline thêm dấu: mỗi từ không dấu thay bằng dạng có dấu hay gặp nhất trong train (giữ hoa/thường như đầu vào).
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
tr, te = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
dem = defaultdict(Counter)
for a, b in zip(tr.text_khong_dau, tr.text):
    for x, y in zip(a.split(), b.split()):
        dem[x][y] += 1
tot = {x: c.most_common(1)[0][0] for x, c in dem.items()}
pd.DataFrame({"id": te.id, "text": [" ".join(tot.get(x, x) for x in s.split()) for s in te.text_khong_dau]}).to_csv("submission.csv", index=False)
print("xong")
