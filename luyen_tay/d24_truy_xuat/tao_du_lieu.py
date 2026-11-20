"""Tạo dữ liệu bài 24: mỗi "đối tượng" = hình chính có sọc + hình phụ ở vị trí riêng, màu riêng.
Train: 100 đối tượng x 6 ảnh. Test: 60 đối tượng mới, mỗi đối tượng 2 ảnh gallery + 2 ảnh query. CPU khoảng 10 giây."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from hinh import ve, nen, lam_hong, luu  # noqa: E402

rng = np.random.default_rng(2401)
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "train", D / "gallery", D / "query", A):
    p.mkdir(parents=True, exist_ok=True)
S = 64


def doi_tuong():
    """Ảnh mẫu (RGB) + mặt nạ của một đối tượng, đặt ở tâm."""
    k1, k2 = rng.integers(6, size=2)
    m1 = ve(int(k1), rng, r=15, tam=(32, 32))[1]
    a = rng.uniform(0, 2 * np.pi)
    m2 = ve(int(k2), rng, r=6, tam=(32 + 17 * np.cos(a), 32 + 17 * np.sin(a)))[1]
    yy, xx = np.mgrid[:S, :S]; phi = rng.uniform(0, np.pi)
    soc = ((xx * np.cos(phi) + yy * np.sin(phi)) // rng.integers(3, 7)) % 2 == 0
    c1, c2, c3 = rng.uniform(0, 255, (3, 3))
    img = np.zeros((S, S, 3)); img[m1 > 0] = c1; img[(m1 > 0) & soc] = c3; img[m2 > 0] = c2
    return img, ((m1 > 0) | (m2 > 0)).astype(np.float32)


def chup(mau, mask):
    """Một lần chụp: xoay, co giãn, dịch chuyển, nền mới, rồi làm hỏng nhẹ."""
    M = cv2.getRotationMatrix2D((32, 32), rng.uniform(0, 360), rng.uniform(0.75, 1.05))
    M[:, 2] += rng.uniform(-5, 5, 2)
    im = cv2.warpAffine(mau.astype(np.float32), M, (S, S)); mk = cv2.warpAffine(mask, M, (S, S))[..., None]
    x = nen(rng) * (1 - mk) + (im * rng.uniform(0.85, 1.15, 3)) * mk
    return lam_hong(np.clip(x, 0, 255).astype(np.uint8), rng, manh=0.4)


rows = []
for o in range(100):
    mau, mk = doi_tuong()
    for v in range(6):
        i = f"tr{o:03d}_{v}"; luu(D / "train" / f"{i}.png", chup(mau, mk)); rows.append((i, f"obj{o:03d}"))
pd.DataFrame(rows, columns=["id", "object"]).to_csv(D / "train.csv", index=False)

objs = [doi_tuong() for _ in range(60)]
gal = [(f"g{j:04d}", int(o)) for j, o in enumerate(rng.permutation(np.repeat(np.arange(60), 2)))]
qry = [(f"q{j:04d}", int(o)) for j, o in enumerate(rng.permutation(np.repeat(np.arange(60), 2)))]
for i, o in gal: luu(D / "gallery" / f"{i}.png", chup(*objs[o]))
for i, o in qry: luu(D / "query" / f"{i}.png", chup(*objs[o]))
pd.DataFrame(gal, columns=["id", "object"]).to_csv(A / "gallery.csv", index=False)
pd.DataFrame(qry, columns=["id", "object"]).to_csv(A / "query.csv", index=False)
pd.DataFrame({"id": [i for i, _ in qry], "matches": " ".join(i for i, _ in gal[:10])}).to_csv(D / "sample_submission.csv", index=False)
print("xong: 600 ảnh train (100 đối tượng),", len(gal), "gallery,", len(qry), "query")
