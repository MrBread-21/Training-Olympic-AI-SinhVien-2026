"""Lời giải mẫu cấp 1. Chỉ mở sau khi đã tự làm và chạy kiem_tra.py."""
import re
import numpy as np


def bai01_mean_std_kenh(X):
    X = X.astype(np.float64)
    return X.mean(axis=(0, 2, 3)), X.std(axis=(0, 2, 3))


def bai02_chuan_hoa(X, mean, std):
    mean, std = np.asarray(mean).reshape(1, -1, 1, 1), np.asarray(std).reshape(1, -1, 1, 1)
    return ((X - mean) / std).astype(np.float32)


def bai03_one_hot(y, k):
    out = np.zeros((len(y), k), np.float32)
    out[np.arange(len(y)), y] = 1
    return out


def bai04_softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # trừ max để exp không tràn số
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def bai05_cross_entropy(logits, y):
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))      # log-sum-exp từng hàng
    return float(np.mean(lse - logits[np.arange(len(y)), y]))


def bai06_accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def bai07_ma_tran_nham(y, p, k):
    return np.bincount(np.asarray(y) * k + np.asarray(p), minlength=k * k).reshape(k, k)


def bai08_macro_f1(y, p, k):
    cm = bai07_ma_tran_nham(y, p, k)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)
    return float(f1.mean())


def bai09_auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    rank = np.empty(len(s))
    i = 0
    while i < len(s):                              # nhóm các điểm bằng nhau, gán hạng trung bình
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        rank[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bai10_cosine(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def bai11_khoang_cach(A, B):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


def bai12_knn(Xtr, ytr, Xte, k):
    nb = np.argsort(bai11_khoang_cach(Xte, Xtr), axis=1, kind="stable")[:, :k]
    n_cls = int(ytr.max()) + 1
    return np.array([np.bincount(ytr[r], minlength=n_cls).argmax() for r in nb])   # argmax lấy nhãn nhỏ nhất khi hoà


def bai13_lam_muot(x, w):
    r = w // 2
    return np.convolve(np.pad(x, r, mode="edge"), np.ones(w) / w, mode="valid")


def bai14_tiou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def bai15_jaccard_tu(a, b):
    tach = lambda s: set(re.sub(r"[^\w\s]", " ", s.lower()).split())
    A, B = tach(a), tach(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def bai16_doan_tong_lon_nhat(x):
    best, cur, s0, bi, bj = -np.inf, 0.0, 0, 0, 0
    for t, v in enumerate(x):
        if cur <= 0:                               # tổng trước đó không giúp gì: bắt đầu đoạn mới tại t
            cur, s0 = v, t
        else:
            cur += v
        if cur > best:
            best, bi, bj = cur, s0, t
    return bi, bj
