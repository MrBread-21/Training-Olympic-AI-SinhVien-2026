import torch                                                   # Windows: nạp torch trước pandas
import random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from torch import nn
import torch.nn.functional as F
import json, math, unicodedata
from PIL import Image

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)

def tu(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s).split()

def doc_anh(thu_muc, ids):
    return np.stack([np.asarray(Image.open(Path(thu_muc) / f"{i}.png").convert("RGB")) for i in ids])   # (N, 64, 64, 3) uint8

def ngram(t):
    return Counter(tuple(t[i:i + k]) for k in range(1, 5) for i in range(len(t) - k + 1))

def cider_d(hyps, refs, sigma=6.0):
    rc = [[ngram(tu(r)) for r in rs] for rs in refs]
    df = Counter(g for cs in rc for g in set().union(*cs)); log_n = math.log(len(refs))
    def vec(c):
        v, n, dai = [{} for _ in range(4)], [0.0] * 4, 0
        for g, tf in c.items():
            k = len(g) - 1; v[k][g] = tf * (log_n - math.log(max(1.0, df[g]))); n[k] += v[k][g] ** 2
            dai += tf if k == 1 else 0
        return v, [math.sqrt(x) for x in n], dai
    out = []
    for h, cs in zip(hyps, rc):
        vh, nh, lh = vec(ngram(tu(h))); s = np.zeros(4)
        for c in cs:
            vr, nr, lr = vec(c)
            for k in range(4):
                val = sum(min(x, vr[k].get(g, 0)) * vr[k].get(g, 0) for g, x in vh[k].items())
                s[k] += (val / (nh[k] * nr[k]) if nh[k] and nr[k] else val) * math.exp(-((lh - lr) ** 2) / (2 * sigma ** 2))
        out.append(s.mean() / len(cs) * 10)
    return float(np.mean(out))

def doc_caption(p):
    return json.load(open(p, encoding="utf-8"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"

cap = doc_caption("data/train_captions.json"); ids = sorted(cap)
te_ids = sorted(p.stem for p in Path("data/test").glob("*.png"))
X = torch.tensor(doc_anh("data/train", ids)).permute(0, 3, 1, 2).float().div(255).to(DEV)
Xt = torch.tensor(doc_anh("data/test", te_ids)).permute(0, 3, 1, 2).float().div(255).to(DEV)

# 1. học đặc trưng ảnh bằng nhãn tự tạo: mỗi ảnh dự đoán tập "từ khái niệm" có trong 5 câu của nó (multi-label, BCE)
KN = sorted({w for c in cap.values() for s in c for w in tu(s)} - {"một", "và", "hình", "màu", "có", "bức", "ảnh", "là", "gồm", "hai", "nằm"})
Y = torch.tensor([[float(any(w in tu(s) for s in cap[i])) for w in KN] for i in ids], device=DEV)
def khoi(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True), nn.MaxPool2d(2))
than = nn.Sequential(khoi(3, 32), khoi(32, 64), khoi(64, 128), khoi(128, 128), nn.Flatten()).to(DEV)   # 128 x 4 x 4: giữ vị trí
dau = nn.Linear(2048, len(KN)).to(DEV)
opt = torch.optim.AdamW(list(than.parameters()) + list(dau.parameters()), lr=2e-3)
for ep in range(15):
    than.train()
    for idx in torch.randperm(len(X), device=DEV).split(64):
        loss = F.binary_cross_entropy_with_logits(dau(than(X[idx])), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
than.eval()
with torch.inference_mode():
    Ftr, Fte = (F.normalize(torch.cat([than(x) for x in Z.split(256)]), dim=1).cpu().numpy() for Z in (X, Xt))

# 2. truy hồi K ảnh gần nhất + chọn câu đồng thuận (MBR theo cosine TF-IDF n-gram, IDF trên caption train)
df = Counter(g for i in ids for g in set().union(*[ngram(tu(s)) for s in cap[i]])); log_n = math.log(len(ids))
def vec(s):
    c = ngram(tu(s)); v = {g: tf * (log_n - math.log(max(1.0, df[g]))) for g, tf in c.items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {g: x / n for g, x in v.items()}
def mbr(ung, w):
    V = [vec(s) for s in ung]
    diem = [sum(wj * sum(a.get(g, 0) * x for g, x in b.items()) for j, (b, wj) in enumerate(zip(V, w)) if j != i) for i, a in enumerate(V)]
    return ung[int(np.argmax(diem))]
def du_doan(Fq, Fdb, db_ids, K):
    S = Fq @ Fdb.T; out = []
    for s in S:
        nb = np.argsort(-s)[:K]
        out.append(mbr([c for j in nb for c in cap[db_ids[j]]], [s[j] for j in nb for _ in cap[db_ids[j]]]))
    return out

val = np.random.default_rng(0).permutation(len(ids))[:300]; con = np.setdiff1d(np.arange(len(ids)), val)
for K in (5, 15, 30):
    p = du_doan(Ftr[val], Ftr[con], [ids[j] for j in con], K)
    print(f"K={K}: val CIDEr-D = {cider_d(p, [cap[ids[j]] for j in val]):.3f}")
pd.DataFrame({"id": te_ids, "caption": du_doan(Fte, Ftr, ids, 15)}).to_csv("submission_a.csv", index=False)
print("đã ghi submission_a.csv")
