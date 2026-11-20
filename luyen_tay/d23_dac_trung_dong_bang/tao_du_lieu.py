"""Tạo dữ liệu bài 23: ảnh theo "phiên chụp" (nhóm 8 ảnh cùng nền, cùng màu, cùng cỡ vật thể).
Train: 96 nhóm (768 ảnh) có cột group; test: 36 nhóm mới (288 ảnh). CPU khoảng 5 giây."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from hinh import LOP, ve, lam_hong, nen, luu  # noqa: E402

rng = np.random.default_rng(2301)
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "train", D / "test", A):
    p.mkdir(parents=True, exist_ok=True)


def phien(n_nhom, tien_to, thu_muc):
    rows, n = [], 0
    for g in range(n_nhom):
        k, bg, r = g % 6, nen(rng), rng.uniform(7, 13)
        mau = rng.uniform(0, 255, 3)
        while np.abs(mau - bg.mean((0, 1))).sum() < 120:
            mau = rng.uniform(0, 255, 3)
        for _ in range(8):
            img = lam_hong(ve(k, rng, r=r * rng.uniform(0.9, 1.1), bg=bg, mau=mau)[0], rng, manh=0.5)
            i = f"{tien_to}{n:04d}"; n += 1
            luu(thu_muc / f"{i}.png", img); rows.append((i, f"{tien_to}g{g:03d}", LOP[k]))
    return pd.DataFrame(rows, columns=["id", "group", "label"])


tr = phien(96, "tr", D / "train")
tr.sample(frac=1, random_state=0).to_csv(D / "train.csv", index=False)
te = phien(36, "te", D / "test")
te[["id", "label"]].to_csv(A / "test.csv", index=False)
te[["id"]].assign(label="tron").sample(frac=1, random_state=1).to_csv(D / "sample_submission.csv", index=False)
print("xong:", len(tr), "ảnh train /", tr.group.nunique(), "nhóm;", len(te), "ảnh test")
