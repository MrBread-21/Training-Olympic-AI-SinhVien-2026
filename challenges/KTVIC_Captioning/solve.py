"""Lời giải KTVIC_Captioning: đặc trưng ảnh từ backbone ImageNet trong danh sách BTC (ConvNeXt-Tiny và ResNet-50, đóng băng,
TTA lật), truy hồi K ảnh train gần nhất, chọn câu đồng thuận bằng MBR (cosine TF-IDF n-gram 1-4, IDF trên caption train,
trọng số theo độ giống ảnh). Phần ngôn ngữ không dùng model nào. K chọn trên 400 ảnh train để riêng.
Chạy: python solve.py <thư_mục_dữ_liệu> (và private_test/ nếu đã giải nén)."""
import torch                                                   # Windows: nạp torch trước pandas
import json, math, random, sys, time, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
import torchvision
from PIL import Image

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
t0 = time.time()
cap = json.load(open(D / "train_captions.json", encoding="utf-8")); ids = sorted(cap)
anh_te = sorted((D / "test/images").glob("*.jpg"))
for p in [Path("private_test/images"), D / "private_test/images"]:
    if p.is_dir(): anh_te += sorted(p.glob("*.jpg")); break

def tu(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s).split()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
mo_hinh = []
r = torchvision.models.resnet50(weights="DEFAULT"); r.fc = torch.nn.Identity(); mo_hinh.append(r.eval().to(DEV))
c = torchvision.models.convnext_tiny(weights="DEFAULT"); c.classifier[2] = torch.nn.Identity(); mo_hinh.append(c.eval().to(DEV))

@torch.inference_mode()
def dac_trung(paths, BS=32):
    out = []
    for i in range(0, len(paths), BS):
        x = torch.stack([torch.tensor(np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC))) for p in paths[i:i + BS]])
        x = (x.permute(0, 3, 1, 2).float().div(255).to(DEV) - MEAN) / STD
        f = [torch.nn.functional.normalize(m(x) + m(x.flip(3)), dim=1) for m in mo_hinh]      # TTA lật ngang
        out.append(torch.nn.functional.normalize(torch.cat(f, 1), dim=1).cpu())
    return torch.cat(out).numpy()

Ftr = dac_trung([D / "train/images" / f"{i}.jpg" for i in ids]); Fte = dac_trung(anh_te)
print(f"đặc trưng: train {Ftr.shape}, test {Fte.shape}, {time.time() - t0:.0f}s")

ngram = lambda t: Counter(tuple(t[i:i + k]) for k in range(1, 5) for i in range(len(t) - k + 1))
df = Counter(g for i in ids for g in set().union(*[ngram(tu(s)) for s in cap[i]])); log_n = math.log(len(ids))
VEC = {}
def vec(s):
    if s not in VEC:
        v = {g: tf * (log_n - math.log(max(1.0, df[g]))) for g, tf in ngram(tu(s)).items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        VEC[s] = {g: x / n for g, x in v.items()}
    return VEC[s]

def mbr(ung, w):
    V = [vec(s) for s in ung]
    diem = [sum(wj * sum(a.get(g, 0) * x for g, x in b.items()) for j, (b, wj) in enumerate(zip(V, w)) if j != i) for i, a in enumerate(V)]
    return ung[int(np.argmax(diem))]

def du_doan(Fq, Fdb, db_ids, K):
    S = Fq @ Fdb.T; out = []
    for s in S:
        nb = np.argsort(-s)[:K]
        out.append(mbr([x for j in nb for x in cap[db_ids[j]]], [float(s[j]) ** 3 for j in nb for _ in cap[db_ids[j]]]))
    return out

def cider_d(hyps, refs):
    rc = [[ngram(tu(r)) for r in rs] for rs in refs]; dfx = Counter(g for cs in rc for g in set().union(*cs)); ln = math.log(len(refs))
    def v(c):
        vv, n, dai = [{} for _ in range(4)], [0.0] * 4, 0
        for g, tf in c.items():
            k = len(g) - 1; vv[k][g] = tf * (ln - math.log(max(1.0, dfx[g]))); n[k] += vv[k][g] ** 2; dai += tf if k == 1 else 0
        return vv, [math.sqrt(x) for x in n], dai
    out = []
    for h, cs in zip(hyps, rc):
        vh, nh, lh = v(ngram(tu(h))); s = np.zeros(4)
        for cc in cs:
            vr, nr, lr = v(cc)
            for k in range(4):
                val = sum(min(x, vr[k].get(g, 0)) * vr[k].get(g, 0) for g, x in vh[k].items())
                s[k] += (val / (nh[k] * nr[k]) if nh[k] and nr[k] else val) * math.exp(-((lh - lr) ** 2) / 72)
        out.append(s.mean() / len(cs) * 10)
    return float(np.mean(out))

val = np.random.default_rng(0).permutation(len(ids))[:400]; con = np.setdiff1d(np.arange(len(ids)), val)
kq = {K: cider_d(du_doan(Ftr[val], Ftr[con], [ids[j] for j in con], K), [cap[ids[j]] for j in val]) for K in (5, 15, 30, 60)}
K = max(kq, key=kq.get); print("val CIDEr-D theo K:", {k: round(v, 3) for k, v in kq.items()}, f"chọn K = {K}, {time.time() - t0:.0f}s")
pd.DataFrame({"id": [p.stem for p in anh_te], "caption": du_doan(Fte, Ftr, ids, K)}).to_csv("submission.csv", index=False)
print(f"đã ghi submission.csv: {len(anh_te)} ảnh, {time.time() - t0:.0f}s")
