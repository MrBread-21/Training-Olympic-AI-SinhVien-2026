import torch                                                   # Windows: nạp torch trước pandas
import random
from pathlib import Path
import numpy as np, pandas as pd
from torch import nn
import torch.nn.functional as F
import json, math, unicodedata
from collections import Counter
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
itos = ["<pad>", "<bos>", "<eos>"] + sorted({w for c in cap.values() for s in c for w in tu(s)}); stoi = {w: i for i, w in enumerate(itos)}
L = 2 + max(len(tu(s)) for c in cap.values() for s in c)
def ma(s):
    t = [1] + [stoi[w] for w in tu(s)] + [2]
    return t + [0] * (L - len(t))
T = torch.tensor([[ma(s) for s in cap[i]] for i in ids], device=DEV)          # (N, 5, L)

class CapModel(nn.Module):
    def __init__(self, V, d=256):
        super().__init__()
        def khoi(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True), nn.MaxPool2d(2))
        self.enc = nn.Sequential(khoi(3, 32), khoi(32, 64), khoi(64, 128), khoi(128, 128), nn.Flatten(), nn.Linear(2048, d), nn.Tanh())
        self.emb = nn.Embedding(V, d, padding_idx=0); self.gru = nn.GRU(2 * d, d, batch_first=True)
        self.drop = nn.Dropout(0.2); self.out = nn.Linear(d, V)
    def forward(self, x, tok):                                   # tok: (B, L) đầu vào decoder; ảnh làm h0 và ghép vào mỗi bước
        f = self.enc(x)
        inp = torch.cat([self.drop(self.emb(tok)), f.unsqueeze(1).expand(-1, tok.shape[1], -1)], -1)
        o, _ = self.gru(inp, f.unsqueeze(0).contiguous())
        return self.out(self.drop(o))
    @torch.inference_mode()
    def greedy(self, x, maxlen=L):
        f = self.enc(x); h = f.unsqueeze(0).contiguous(); tok = torch.ones(len(x), 1, dtype=torch.long, device=x.device); out = []
        for _ in range(maxlen):
            o, h = self.gru(torch.cat([self.emb(tok[:, -1:]), f.unsqueeze(1)], -1), h)
            nxt = self.out(o[:, -1]).argmax(-1, keepdim=True); out.append(nxt); tok = torch.cat([tok, nxt], 1)
        return torch.cat(out, 1).tolist()

def giai_ma(seq):
    w = []
    for t in seq:
        if t == 2: break
        if t > 2: w.append(itos[t])
    return " ".join(w)

def train(idx_train, EP=40, BS=64):
    m = CapModel(len(itos)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EP * ((len(idx_train) + BS - 1) // BS))
    for ep in range(EP):
        m.train()
        for b in idx_train[torch.randperm(len(idx_train), device=DEV)].split(BS):
            t = T[b, torch.randint(0, 5, (len(b),), device=DEV)]            # mỗi lần chọn ngẫu nhiên 1 trong 5 câu
            xb = X[b]
            if random.random() < 0.5:                                         # đổi màu nền nhẹ (không lật: lật làm sai trái / phải)
                xb = (xb + 0.05 * torch.randn(len(b), 3, 1, 1, device=DEV)).clamp(0, 1)
            loss = F.cross_entropy(m(xb, t[:, :-1]).flatten(0, 1), t[:, 1:].flatten(), ignore_index=0, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    return m

def du_doan(m, Z):
    return [giai_ma(s) for z in Z.split(256) for s in m.greedy(z)]

val = torch.tensor(np.random.default_rng(0).permutation(len(ids))[:300], device=DEV)
con = torch.tensor(np.setdiff1d(np.arange(len(ids)), val.cpu().numpy()), device=DEV)
m = train(con)
p = du_doan(m, X[val])
print(f"val CIDEr-D = {cider_d(p, [cap[ids[j]] for j in val.tolist()]):.3f}   ví dụ: {p[0]}  |  {cap[ids[int(val[0])]][0]}")
# (train lại trên toàn bộ train được thêm chút điểm nhưng gấp đôi thời gian; ở đây dùng luôn mô hình 90%)
pd.DataFrame({"id": te_ids, "caption": du_doan(m, Xt)}).to_csv("submission_b.csv", index=False)
print("đã ghi submission_b.csv")
