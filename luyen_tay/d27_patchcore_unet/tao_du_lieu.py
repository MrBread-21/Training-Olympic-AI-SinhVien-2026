"""Tạo dữ liệu bài 27: bề mặt vân sọc 64x64; lỗi = vết xước, đốm, mảng mất vân.
Phần A: 200 ảnh train bình thường, 200 ảnh test (một nửa có lỗi). Phần B: 400 ảnh lỗi có mặt nạ, 100 ảnh test.
CPU khoảng 5 giây."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(2701)
D, A = HERE / "data", HERE / ".dap_an"
for p in ("a_train", "a_test", "b_train/images", "b_train/masks", "b_test"):
    (D / p).mkdir(parents=True, exist_ok=True)
(A / "b_masks").mkdir(parents=True, exist_ok=True)
S = 64
yy, xx = np.mgrid[:S, :S].astype(np.float32)


def be_mat():
    th = np.radians(30 + rng.normal(0, 4)); f = 0.16 + rng.normal(0, 0.008)
    v = 128 + 45 * np.sin(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)) + rng.uniform(0, 2 * np.pi))
    img = v[..., None] * np.array([0.9, 0.75, 0.55]) * rng.uniform(0.95, 1.05) + 30
    return img + rng.normal(0, 6, img.shape)


def loi(img, nhe=False):
    """nhe=True: lỗi mờ hơn (trộn 35-60%) và nhỏ hơn, dùng cho test phần A."""
    m = np.zeros((S, S), np.uint8); kieu = rng.integers(3); goc = img.copy(); sc = 0.6 if nhe else 1.0
    if kieu == 0:                                                     # vết xước
        p1, p2 = rng.uniform(8, 56, 2), rng.uniform(8, 56, 2)
        p2 = p1 + (p2 - p1) * sc
        cv2.line(m, tuple(p1.astype(int)), tuple(p2.astype(int)), 1, int(rng.integers(1, 3) if nhe else rng.integers(2, 4)))
        img = np.where(m[..., None] > 0, rng.uniform(20, 60), img)
    elif kieu == 1:                                                   # đốm
        cv2.ellipse(m, tuple(rng.uniform(12, 52, 2).astype(int)), tuple((rng.uniform(3, 8, 2) * sc).astype(int)),
                    rng.uniform(0, 180), 0, 360, 1, -1)
        img = np.where(m[..., None] > 0, rng.uniform(0, 255, 3), img)
    else:                                                             # mảng mất vân
        x0, y0 = rng.integers(6, 40, 2); w, h = (rng.integers(8, 18, 2) * sc).astype(int)
        m[y0:y0 + h, x0:x0 + w] = 1
        img = np.where(m[..., None] > 0, img.mean((0, 1)) + rng.normal(0, 6, img.shape), img)
    if nhe:
        a = rng.uniform(0.35, 0.6); img = goc * (1 - a) + img * a
    return img, m


def luu(p, img): cv2.imwrite(str(p), np.clip(img, 0, 255).astype(np.uint8)[..., ::-1])


for i in range(200): luu(D / "a_train" / f"n{i:03d}.png", be_mat())
rows = []
for j, i in enumerate(rng.permutation(200)):
    co = int(j < 100); img = be_mat()
    if co: img = loi(img, nhe=True)[0]
    luu(D / "a_test" / f"a{i:03d}.png", img); rows.append((f"a{i:03d}", co))
pd.DataFrame(rows, columns=["id", "anomaly"]).sort_values("id").to_csv(A / "a_test.csv", index=False)
for i in range(400):
    img, m = loi(be_mat()); luu(D / "b_train/images" / f"s{i:03d}.png", img); cv2.imwrite(str(D / "b_train/masks" / f"s{i:03d}.png"), m * 255)
for i in range(100):
    img, m = loi(be_mat()); luu(D / "b_test" / f"b{i:03d}.png", img); cv2.imwrite(str(A / "b_masks" / f"b{i:03d}.png"), m * 255)
print("xong: phần A 200 train / 200 test, phần B 400 train / 100 test")
