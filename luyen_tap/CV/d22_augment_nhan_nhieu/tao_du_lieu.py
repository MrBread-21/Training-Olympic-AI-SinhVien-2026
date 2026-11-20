"""Tạo dữ liệu bài 22: 1800 ảnh train sạch nhưng khoảng 35% nhãn sai; 600 ảnh test bị làm hỏng mạnh. CPU khoảng 10 giây."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from hinh import LOP, ve, lam_hong, luu  # noqa: E402

rng = np.random.default_rng(2201)
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "train", D / "test", A):
    p.mkdir(parents=True, exist_ok=True)

rows, sai = [], 0
for i in range(1800):
    k = i % 6
    img, _ = ve(k, rng)
    luu(D / "train" / f"tr{i:04d}.png", img)
    nhan = k
    if rng.random() < 0.35:                                       # nhãn nhiễu đối xứng: đổi sang một lớp khác
        nhan = int((k + rng.integers(1, 6)) % 6); sai += 1
    rows.append((f"tr{i:04d}", LOP[nhan], LOP[k]))
tr = pd.DataFrame(rows, columns=["id", "label", "dung"]).sample(frac=1, random_state=0)
tr[["id", "label"]].to_csv(D / "train.csv", index=False)
tr[["id", "dung"]].to_csv(A / "train_dung.csv", index=False)

rows = []
for i in rng.permutation(600):
    k = int(rng.integers(6))
    luu(D / "test" / f"te{i:04d}.png", lam_hong(ve(k, rng)[0], rng, manh=1.0)); rows.append((f"te{i:04d}", LOP[k]))
ans = pd.DataFrame(rows, columns=["id", "label"]).sort_values("id")
ans.to_csv(A / "test.csv", index=False)
ans.assign(label="tron").sample(frac=1, random_state=1).to_csv(D / "sample_submission.csv", index=False)
print(f"xong: 1800 ảnh train ({sai} nhãn sai), {len(ans)} ảnh test")
