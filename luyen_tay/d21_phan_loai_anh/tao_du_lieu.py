"""Tạo dữ liệu bài 21: 1200 ảnh train có nhãn, 600 ảnh test (biến đổi nhẹ). Chạy CPU khoảng 10 giây."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from hinh import LOP, ve, lam_hong, luu  # noqa: E402

rng = np.random.default_rng(2101)
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "train", D / "test", A):
    p.mkdir(parents=True, exist_ok=True)

rows = []
for i in range(1200):
    k = i % 6
    img, _ = ve(k, rng)
    luu(D / "train" / f"tr{i:04d}.png", img); rows.append((f"tr{i:04d}", LOP[k]))
pd.DataFrame(rows, columns=["id", "label"]).sample(frac=1, random_state=0).to_csv(D / "train.csv", index=False)

rows = []
ids = rng.permutation(600)
for j, i in enumerate(ids):
    k = int(rng.integers(6))
    img = lam_hong(ve(k, rng)[0], rng, manh=0.3)
    luu(D / "test" / f"te{i:04d}.png", img); rows.append((f"te{i:04d}", LOP[k]))
ans = pd.DataFrame(rows, columns=["id", "label"]).sort_values("id")
ans.to_csv(A / "test.csv", index=False)
ans.assign(label="tron").sample(frac=1, random_state=1).to_csv(D / "sample_submission.csv", index=False)
print("xong:", len(list((D / "train").iterdir())), "ảnh train,", len(ans), "ảnh test")
