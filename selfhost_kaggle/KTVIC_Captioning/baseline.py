# %% [markdown]
# # Baseline - Sinh caption tiếng Việt cho ảnh (KTVIC)
#
# Không học mô hình, không pretrained: mô tả mỗi ảnh bằng ảnh thu nhỏ 16x16 và histogram màu, tìm K ảnh train gần nhất,
# gom các câu mô tả của chúng rồi chọn câu "đồng thuận" nhất (giống các câu khác nhất theo từ và cặp từ).
# Tự chấm CIDEr-D trên 300 ảnh train để riêng, rồi ghi `submission.csv` (`id,caption`).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("train_captions.json")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)


def dac_trung(p):
    im = Image.open(p).convert("RGB")
    nho = np.asarray(im.resize((16, 16), Image.BILINEAR), np.float32).ravel() / 255
    hsv = np.asarray(im.resize((64, 64)).convert("HSV"))
    hist = np.concatenate([np.histogram(hsv[..., k], 16, (0, 256))[0] for k in range(3)]).astype(np.float32) / 4096
    v = np.concatenate([nho - nho.mean(), 3 * hist])
    return v / (np.linalg.norm(v) + 1e-8)


def tu(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s).split()


def dong_thuan(caps):
    """Câu có tổng độ trùng (từ + cặp từ) với các câu khác lớn nhất."""
    bo = [Counter(t) + Counter(zip(t, t[1:])) for t in map(tu, caps)]
    diem = [sum(sum((a & b).values()) / (1 + sum(b.values())) for j, b in enumerate(bo) if j != i) for i, a in enumerate(bo)]
    return caps[int(np.argmax(diem))]


def cider_d(hyp, refs):
    """CIDEr-D giống metric của cuộc thi. hyp: list câu, refs: list (list câu)."""
    dem = lambda t: Counter(tuple(t[i:i + k]) for k in range(1, 5) for i in range(len(t) - k + 1))
    rc = [[dem(tu(r)) for r in rs] for rs in refs]
    df = Counter(g for cs in rc for g in set().union(*cs))
    logn = math.log(len(refs))

    def vec(c):
        v, n, dai = [{} for _ in range(4)], [0.0] * 4, 0
        for g, tf in c.items():
            k = len(g) - 1; v[k][g] = tf * (logn - math.log(max(1.0, df[g]))); n[k] += v[k][g] ** 2
            dai += tf if k == 1 else 0
        return v, [math.sqrt(x) for x in n], dai

    out = []
    for h, cs in zip(hyp, rc):
        vh, nh, lh = vec(dem(tu(h))); s = np.zeros(4)
        for c in cs:
            vr, nr, lr = vec(c)
            for k in range(4):
                val = sum(min(x, vr[k].get(g, 0)) * vr[k].get(g, 0) for g, x in vh[k].items())
                s[k] += (val / (nh[k] * nr[k]) if nh[k] and nr[k] else val) * math.exp(-((lh - lr) ** 2) / 72)
        out.append(s.mean() / len(cs) * 10)
    return float(np.mean(out))


cap = json.load(open(D / "train_captions.json", encoding="utf-8"))
ids = sorted(cap)
F = np.stack([dac_trung(D / "train/images" / f"{i}.jpg") for i in ids])
print("đặc trưng train:", F.shape)


def du_doan(Fq, Fdb, ids_db, K=10):
    S = Fq @ Fdb.T
    return [dong_thuan([c for j in np.argsort(-s)[:K] for c in cap[ids_db[j]]]) for s in S]


# %% [markdown]
# ## Tự chấm trên 300 ảnh train để riêng

# %%
rng = np.random.default_rng(42)
val = rng.permutation(len(ids))[:300]; con = np.setdiff1d(np.arange(len(ids)), val)
for K in [10, 25, 50]:
    p = du_doan(F[val], F[con], [ids[j] for j in con], K)
    print(f"K={K}: val CIDEr-D = {cider_d(p, [cap[ids[j]] for j in val]):.4f}")

# %% [markdown]
# ## Dự đoán test (public) và private_test (nếu đã giải nén)

# %%
anh = sorted((D / "test/images").glob("*.jpg"))
for thu in [Path("private_test/images"), D / "private_test/images"]:
    if thu.is_dir():
        anh += sorted(thu.glob("*.jpg")); break
p = du_doan(np.stack([dac_trung(a) for a in anh]), F, ids, 10)
pd.DataFrame({"id": [a.stem for a in anh], "caption": p}).to_csv("submission.csv", index=False)
print(f"ghi submission.csv: {len(anh)} ảnh")
