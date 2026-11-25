"""Baseline Nordland: ảnh thu nhỏ 64x32 xám, chuẩn hoá từng ô, so khớp láng giềng gần nhất (cosine).
Không học gì; là mức điểm tối thiểu.  Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")


def vec(p):
    a = np.asarray(Image.open(p).convert("L").resize((64, 32), Image.BILINEAR), dtype=np.float32)
    a = a.reshape(4, 8, 8, 8).transpose(0, 2, 1, 3).reshape(32, -1)  # 32 ô 8x8
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    v = a.flatten()
    return v / np.linalg.norm(v)


q = pd.read_csv(D / "query.csv"); db = pd.read_csv(D / "database.csv")
Q = np.stack([vec(D / f"test/query/{i}.jpg") for i in q.id])
B = np.stack([vec(D / f"test/database/{i}.jpg") for i in db.id])
top = np.argsort(-(Q @ B.T), axis=1)[:, :10]
pd.DataFrame({"id": q.id, "predictions": [" ".join(db.id.values[t]) for t in top]}).to_csv("submission.csv", index=False)
print("xong", len(q))
