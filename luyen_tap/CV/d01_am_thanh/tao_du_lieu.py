"""Tạo dữ liệu bài 01 (âm thanh tổng hợp, CPU khoảng 1 phút): "từ khoá" là chuỗi 1-3 âm tiết, mỗi âm tiết có nguyên âm
(cặp formant) và đường cao độ (bằng / lên / xuống / gãy) riêng. Người nói khác nhau về cao độ gốc, tốc độ nói, âm sắc.
8 từ khoá + lớp "khac" (các tổ hợp âm tiết khác). File .wav 16 kHz, 1 giây, int16.
Train: 1.800 đoạn âm thanh sạch của 40 người nói. Test: 900 đoạn của 20 người nói khác, 80% bị trộn tạp âm
(nhiễu trắng, nhiễu hồng, tiếng nói khác) với SNR 0-15 dB, lệch thời gian, đổi âm lượng."""
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "train", D / "test", A):
    p.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(101); R = random.Random(101)
SR = 16000
NGUYEN_AM = {"a": (750, 1200), "i": (300, 2300), "u": (320, 800), "e": (500, 1900), "o": (500, 900)}
CAO_DO = {"bang": lambda x: np.ones_like(x), "len": lambda x: 0.9 + 0.35 * x, "xuong": lambda x: 1.25 - 0.4 * x,
          "gay": lambda x: 1.0 - 0.3 * np.sin(np.pi * x)}
TU = {"len": [("e", "len")], "xuong": [("u", "xuong"), ("o", "bang")], "trai": [("a", "bang"), ("i", "len")],
      "phai": [("a", "xuong"), ("i", "bang")], "dung": [("u", "gay")], "di": [("i", "xuong")],
      "bat": [("a", "len"), ("a", "len")], "tat": [("o", "len"), ("e", "xuong"), ("a", "bang")]}
KHOA = {tuple(v) for v in TU.values()}
AM_TIET = list(itertools.product(NGUYEN_AM, CAO_DO))


def tu_khac():
    while True:
        w = tuple(R.choice(AM_TIET) for _ in range(R.choice([1, 2, 2, 3])))
        if w not in KHOA:
            return list(w)


def nguoi_noi():
    return dict(f0=rng.uniform(95, 260), toc_do=rng.uniform(0.8, 1.2), am_sac=rng.uniform(0.88, 1.12))


def doc_tu(am_tiet, nn):
    out = []
    for na, cd in am_tiet:
        d = rng.uniform(0.15, 0.22) / nn["toc_do"]; n = int(d * SR)
        x = np.linspace(0, 1, n)
        f0 = nn["f0"] * CAO_DO[cd](x) * rng.uniform(0.97, 1.03)
        pha = 2 * np.pi * np.cumsum(f0) / SR
        F1, F2 = (f * nn["am_sac"] for f in NGUYEN_AM[na])
        k = np.arange(1, int(4000 / nn["f0"]) + 1)[:, None]
        hz = k * f0[None, :]
        bien_do = np.exp(-((hz - F1) ** 2) / (2 * 120 ** 2)) + 0.7 * np.exp(-((hz - F2) ** 2) / (2 * 180 ** 2)) + 0.02
        s = (bien_do * np.sin(k * pha[None, :])).sum(0)
        env = np.minimum(1, np.minimum(x, 1 - x) * 12)
        out += [s * env, np.zeros(int(rng.uniform(0.03, 0.08) * SR))]
    return np.concatenate(out)


def dat_vao_1_giay(s):
    y = np.zeros(SR)
    s = s[:SR - 400]; bd = int(rng.integers(200, SR - len(s) - 199))
    y[bd:bd + len(s)] = s
    return y / (np.abs(y).max() + 1e-9) * rng.uniform(0.3, 0.9)


def nhieu(y, kho):
    loai = rng.choice(["trang", "hong", "noi"])
    n = rng.standard_normal(SR)
    if loai == "hong":
        f = np.fft.rfftfreq(SR); f[0] = f[1]; n = np.fft.irfft(np.fft.rfft(n) / np.sqrt(f), SR)
    elif loai == "noi":
        n = sum(kho[i] for i in rng.choice(len(kho), 3, replace=False))
    snr = rng.uniform(0, 15)
    y = y + n * np.sqrt((np.mean(y ** 2) + 1e-9) / (np.mean(n ** 2) + 1e-9) / 10 ** (snr / 10))
    return y / (np.abs(y).max() + 1e-9) * rng.uniform(0.2, 0.9)


def tao(n, so_nguoi, tien_to, thu_muc, lam_nhieu, kho=None):
    nguoi = [nguoi_noi() for _ in range(so_nguoi)]; rows = []
    lop = list(TU) + ["khac"]
    for i in range(n):
        nhan = lop[i % len(lop)] if R.random() < 0.75 else "khac"
        nn = nguoi[R.randrange(so_nguoi)]
        y = dat_vao_1_giay(doc_tu(TU[nhan] if nhan != "khac" else tu_khac(), nn))
        if lam_nhieu and R.random() < 0.8:
            y = nhieu(y, kho)
        i_ = f"{tien_to}{i:05d}"
        wavfile.write(thu_muc / f"{i_}.wav", SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
        rows.append((i_, nhan, f"spk{nguoi.index(nn):02d}"))
    return pd.DataFrame(rows, columns=["id", "label", "speaker"])


tr = tao(1800, 40, "tr", D / "train", False)
kho = [dat_vao_1_giay(doc_tu(tu_khac(), nguoi_noi())) for _ in range(60)]          # tiếng nói nền cho test
te = tao(900, 20, "te", D / "test", True, kho)
tr.to_csv(D / "train.csv", index=False)
te[["id"]].to_csv(D / "test.csv", index=False)
te[["id", "label"]].sort_values("id").to_csv(A / "test.csv", index=False)
print(f"xong: train {len(tr)} ({tr.label.value_counts().to_dict()}), test {len(te)}")
