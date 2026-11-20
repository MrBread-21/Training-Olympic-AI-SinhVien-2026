"""Chấm bài cấp 1.
  python kiem_tra.py              chấm mọi bài trong bai_tap.py
  python kiem_tra.py 5 9          chỉ chấm bài 5 và 9
  python kiem_tra.py --loi_giai   chấm lời giải mẫu (để kiểm tra bộ chấm)
  python kiem_tra.py viet_lai.py 1 2 3   chấm file khác (bài viết lại nhanh)
Đáp án so sánh lấy từ sklearn / torch / cách tính trực tiếp bằng vòng lặp, độc lập với bài làm."""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--") and not a.endswith(".py")]
files = [a for a in sys.argv[1:] if a.endswith(".py")]
src = Path(files[0]).resolve() if files else HERE / ("loi_giai/loi_giai.py" if "--loi_giai" in sys.argv else "bai_tap.py")
spec = importlib.util.spec_from_file_location("bai", src)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
rng = np.random.default_rng(0)


def close(a, b, tol=1e-6, msg=""):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{msg} sai shape: {a.shape}, cần {b.shape}"
    assert np.allclose(a, b, atol=tol, rtol=tol), f"{msg} sai giá trị, lệch tối đa {np.abs(a - b).max():.3g}"


# ---------------------------------------------------------------- các bài kiểm tra
def t01():
    X = rng.integers(0, 256, (20, 3, 8, 8), dtype=np.uint8)
    m, s = M.bai01_mean_std_kenh(X)
    Xf = X.astype(np.float64)
    close(m, [Xf[:, c].mean() for c in range(3)], msg="mean")
    close(s, [Xf[:, c].std() for c in range(3)], msg="std")


def t02():
    X = rng.integers(0, 256, (5, 3, 4, 4)).astype(np.uint8)
    mean, std = np.array([10., 100., 200.]), np.array([2., 4., 8.])
    out = M.bai02_chuan_hoa(X, mean, std)
    assert np.asarray(out).dtype == np.float32, f"cần float32, đang là {np.asarray(out).dtype}"
    ref = np.stack([(X[:, c].astype(float) - mean[c]) / std[c] for c in range(3)], 1)
    close(out, ref, tol=1e-5)


def t03():
    y = rng.integers(0, 7, 50)
    out = M.bai03_one_hot(y, 7)
    assert np.asarray(out).dtype == np.float32, "cần float32"
    ref = np.zeros((50, 7))
    for i, c in enumerate(y): ref[i, c] = 1
    close(out, ref)


def t04():
    import torch
    for scale in (1, 1000):
        z = rng.normal(size=(30, 6)) * scale
        out = np.asarray(M.bai04_softmax(z))
        assert np.isfinite(out).all(), f"có nan/inf khi giá trị cỡ {scale} (cần trừ max)"
        close(out, torch.softmax(torch.tensor(z), 1).numpy(), msg=f"giá trị cỡ {scale}:")


def t05():
    import torch
    for scale in (1, 1000):
        z, y = rng.normal(size=(40, 5)) * scale, rng.integers(0, 5, 40)
        out = M.bai05_cross_entropy(z, y)
        assert np.isfinite(out), f"ra nan/inf khi logits cỡ {scale} (cần log-sum-exp)"
        ref = torch.nn.functional.cross_entropy(torch.tensor(z), torch.tensor(y)).item()
        close(out, ref, tol=1e-6, msg=f"logits cỡ {scale}:")


def t06():
    y, p = rng.integers(0, 3, 100), rng.integers(0, 3, 100)
    close(M.bai06_accuracy(y, p), sum(a == b for a, b in zip(y, p)) / 100)


def t07():
    from sklearn.metrics import confusion_matrix
    y, p = rng.integers(0, 4, 200), rng.integers(0, 4, 200)
    p[:5] = 5                                                     # có lớp chỉ xuất hiện trong dự đoán
    out = M.bai07_ma_tran_nham(y, p, 6)
    close(out, confusion_matrix(y, p, labels=range(6)))


def t08():
    from sklearn.metrics import f1_score
    for k, n in ((5, 300), (6, 40)):
        y, p = rng.integers(0, k - 1, n), rng.integers(0, k - 1, n)   # lớp k-1 không xuất hiện ở đâu cả
        ref = f1_score(y, p, labels=range(k), average="macro", zero_division=0)
        close(M.bai08_macro_f1(y, p, k), ref, msg=f"k={k}:")


def t09():
    from sklearn.metrics import roc_auc_score
    y = rng.integers(0, 2, 300)
    s = y * 0.8 + rng.normal(size=300)
    close(M.bai09_auroc(y, s), roc_auc_score(y, s), msg="điểm liên tục:")
    s2 = np.round(s)                                              # nhiều điểm bằng nhau
    close(M.bai09_auroc(y, s2), roc_auc_score(y, s2), msg="có điểm bằng nhau (hạng trung bình):")


def t10():
    A, B = rng.normal(size=(6, 4)), rng.normal(size=(9, 4))
    ref = [[a @ b / (np.sqrt(a @ a) * np.sqrt(b @ b)) for b in B] for a in A]
    close(M.bai10_cosine(A, B), ref)


def t11():
    A, B = rng.normal(size=(7, 5)), rng.normal(size=(4, 5))
    A[0] = B[0]                                                   # khoảng cách đúng bằng 0
    out = np.asarray(M.bai11_khoang_cach(A, B))
    assert np.isfinite(out).all(), "có nan (căn của số âm nhỏ?)"
    close(out, [[np.sqrt(((a - b) ** 2).sum()) for b in B] for a in A], tol=1e-6)


def t12():
    Xtr, ytr = rng.normal(size=(60, 3)), rng.integers(0, 4, 60)
    Xte = rng.normal(size=(25, 3))
    for k in (1, 4, 7):
        ref = []
        for x in Xte:
            d = [np.sqrt(((x - xt) ** 2).sum()) for xt in Xtr]
            nb = sorted(range(len(Xtr)), key=lambda i: d[i])[:k]
            votes = [sum(ytr[i] == c for i in nb) for c in range(4)]
            ref.append(votes.index(max(votes)))
        close(M.bai12_knn(Xtr, ytr, Xte, k), ref, msg=f"k={k}:")


def t13():
    x = rng.normal(size=30)
    for w in (1, 3, 7):
        r = w // 2
        ref = [np.mean([x[min(max(t + d, 0), 29)] for d in range(-r, r + 1)]) for t in range(30)]
        close(M.bai13_lam_muot(x, w), ref, msg=f"w={w}:")


def t14():
    cases = [((3, 5), (5, 9), 1 / 7), ((1, 10), (1, 10), 1.0), ((1, 4), (6, 9), 0.0), ((2, 8), (4, 5), 2 / 7), ((5, 5), (5, 5), 1.0)]
    for a, b, ref in cases:
        close(M.bai14_tiou(a, b), ref, msg=f"{a}, {b}:")
        close(M.bai14_tiou(b, a), ref, msg=f"{b}, {a} (đổi thứ tự):")


def t15():
    cases = [("Năm 1911.", "năm 1911", 1.0), ("", "", 1.0), ("a b", "b c", 1 / 3), ("Gyeonggi, The King", "the king", 2 / 3),
             ("15 tháng 12", "", 0.0), ("Hà Nội-Việt Nam", "hà nội", 0.5)]
    for a, b, ref in cases:
        close(M.bai15_jaccard_tu(a, b), ref, msg=f"{a!r} vs {b!r}:")


def t16():
    for T in (1, 5, 50, 200):
        x = rng.normal(size=T)
        best, ref = -np.inf, None
        for i in range(T):
            for j in range(i, T):
                if x[i:j + 1].sum() > best: best, ref = x[i:j + 1].sum(), (i, j)
        out = tuple(int(v) for v in M.bai16_doan_tong_lon_nhat(x))
        assert out == ref, f"T={T}: trả {out}, cần {ref}"
    out = tuple(int(v) for v in M.bai16_doan_tong_lon_nhat(np.array([-3., -1., -2.])))
    assert out == (1, 1), f"toàn số âm: trả {out}, cần (1, 1)"


# ---------------------------------------------------------------- chạy
TESTS = {int(n[1:]): f for n, f in sorted(globals().items()) if n.startswith("t") and n[1:].isdigit()}
chon = [int(a) for a in args] or sorted(TESTS)
ok = 0
print(f"Chấm {src.name}\n")
for n in chon:
    t0 = time.time()
    try:
        TESTS[n]()
        ok += 1; kq = "ĐÚNG"
    except NotImplementedError:
        kq = "chưa làm"
    except AssertionError as e:
        kq = f"SAI   {e}"
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        kq = f"LỖI   {type(e).__name__}: {e} (dòng {tb.lineno})"
    print(f"  bài {n:02d}  {kq}")
print(f"\nĐúng {ok}/{len(chon)}")
