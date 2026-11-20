"""Tạo dữ liệu bài 26: video 64x64, 40-64 khung, 12 fps. 5 loại: binh_thuong và 4 sự kiện xảy ra trong một đoạn khung.
Train 200 video, test 100 video. CPU khoảng 30 giây."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from hinh import nen  # noqa: E402

LOP = ["binh_thuong", "nhay", "quay", "chay_ngang", "nhap_nhay"]
rng = np.random.default_rng(2601)
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "videos", A):
    p.mkdir(parents=True, exist_ok=True)


def video(k, path):
    T = int(rng.integers(40, 65)); bg = nen(rng)
    s = int(rng.integers(6, T - 14)); e = int(min(T, s + rng.integers(10, 30)))           # đoạn sự kiện 1-based [s, e]
    nhieu = rng.uniform(10, 54, (2, 2)); v = rng.normal(0, 0.6, (2, 2))
    mau = rng.uniform(0, 255, 3); cx, cy = rng.uniform(16, 48, 2); ph = rng.uniform(0, 2 * np.pi)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 64))
    for t in range(1, T + 1):
        f = bg + rng.normal(0, 5, bg.shape)
        nhieu = np.clip(nhieu + v, 6, 58)
        for x, y in nhieu: cv2.circle(f, (int(x), int(y)), 4, (90, 90, 90), -1)          # vật thể nền trôi chậm
        if k and s <= t <= e:
            u = (t - s) / 3.0
            if k == 1: cv2.circle(f, (int(cx), int(32 + 20 * np.sin(u * 2.2))), 6, mau, -1)
            elif k == 2: cv2.circle(f, (int(32 + 18 * np.cos(u + ph)), int(32 + 18 * np.sin(u + ph))), 6, mau, -1)
            elif k == 3: cv2.rectangle(f, (int((t - s) * 64 / (e - s + 1)) - 6, int(cy) - 5), (int((t - s) * 64 / (e - s + 1)) + 6, int(cy) + 5), mau, -1)
            elif k == 4 and (t - s) % 4 < 2: cv2.circle(f, (int(cx), int(cy)), 12, mau, -1)
        w.write(np.clip(f, 0, 255).astype(np.uint8)[..., ::-1])
    w.release()
    return T, (s, e) if k else ("", "")


def tao(n, tien_to):
    rows, info = [], []
    for i in range(n):
        k = i % 5; vid = f"{tien_to}{i:03d}"
        T, (s, e) = video(k, D / "videos" / f"{vid}.mp4")
        rows.append((vid, LOP[k], s, e)); info.append((vid, T))
    return pd.DataFrame(rows, columns=["video_id", "label", "start_frame", "end_frame"]), info


tr, i1 = tao(200, "tr")
te, i2 = tao(100, "te")
tr.sample(frac=1, random_state=0).to_csv(D / "train.csv", index=False)
te.to_csv(A / "test.csv", index=False)
pd.DataFrame(i1 + i2, columns=["video_id", "frames"]).to_csv(D / "video_info.csv", index=False)
te[["video_id"]].assign(label="binh_thuong", start_frame="", end_frame="").to_csv(D / "sample_submission.csv", index=False)
print("xong: 200 video train, 100 video test; lớp:", LOP)
