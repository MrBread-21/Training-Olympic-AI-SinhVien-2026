"""Lời giải DimABSA_Quad (Subtask 2, không pretrained): gán nhãn BIO cho aspect / opinion bằng BiLSTM + ký tự (train từ đầu),
ghép mỗi opinion với aspect gần nhất cùng mệnh đề, category bằng logistic, VA bằng Ridge trên opinion + mệnh đề; bỏ bộ tứ NULL
(test không có NULL), khử trùng lặp. Chạy: python solve.py <thư_mục_dữ_liệu> (và private_test.jsonl nếu đã giải nén)."""
import torch                                                   # Windows: nạp torch trước pandas / sklearn
import json, random, re, sys, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from torch import nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
doc = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
train, dev, test = doc(D / "train.jsonl"), doc(D / "dev.jsonl"), doc(D / "test.jsonl")
for p in [Path("private_test.jsonl"), D / "private_test.jsonl"]:
    if p.exists(): test += doc(p); break
TAGS = ["O", "B-ASP", "I-ASP", "B-OPI", "I-OPI"]
NGAT = {",", ".", ";", "!", "?", "but", "and", "though", "although", "however", "while", "whereas", "-", "--"}


def token(text):
    text = unicodedata.normalize("NFC", text)
    return text, [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)]


def chuan(w):
    w = w.lower()
    return {"t": "n't", "nt": "n't"}.get(w, w)


def gan_nhan(r):
    text, sp = token(r["text"]); tag = ["O"] * len(sp); low = text.lower()
    for q in r["quadruplets"]:
        for khoa, loai in (("aspect", "ASP"), ("opinion", "OPI")):
            s = q[khoa].lower()
            if s == "null": continue
            st = -1
            while True:
                st = low.find(s, st + 1)
                if st < 0: break
                idx = [k for k, (a, b) in enumerate(sp) if a >= st and b <= st + len(s)]
                if idx and sp[idx[0]][0] == st and sp[idx[-1]][1] == st + len(s) and all(tag[k] == "O" for k in idx):
                    tag[idx[0]] = "B-" + loai
                    for k in idx[1:]: tag[k] = "I-" + loai
                    break
    return text, sp, tag


mau_tr = [gan_nhan(r) for r in train + dev]           # dev cũng có nhãn: dùng để train bản cuối
mau_dev = [gan_nhan(r) for r in dev]
dem = Counter(chuan(t[a:b]) for t, sp, _ in mau_tr for a, b in sp)
stoi = {w: i + 2 for i, w in enumerate(w for w, c in dem.items() if c >= 2)}
ctoi = {c: i + 2 for i, c in enumerate(sorted({c for t, _, _ in mau_tr for c in t.lower()}))}


def lo(batch, word_drop=0.0):
    L, K = max(len(sp) for _, sp, _ in batch), 14
    W = torch.zeros(len(batch), L, dtype=torch.long); C = torch.zeros(len(batch), L, K, dtype=torch.long)
    Y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (t, sp, tag) in enumerate(batch):
        W[i, :len(sp)] = torch.tensor([1 if random.random() < word_drop else stoi.get(chuan(t[a:b]), 1) for a, b in sp])
        for j, (a, b) in enumerate(sp):
            cc = [ctoi.get(ch, 1) for ch in t[a:b].lower()[:K]]; C[i, j, :len(cc)] = torch.tensor(cc)
        if tag: Y[i, :len(tag)] = torch.tensor([TAGS.index(x) for x in tag])
    return W.to(DEV), C.to(DEV), Y.to(DEV)


class Tagger(nn.Module):
    def __init__(self, Vw, Vc, dw=128, dc=32, h=192):
        super().__init__()
        self.ew = nn.Embedding(Vw, dw, padding_idx=0); self.ec = nn.Embedding(Vc, dc, padding_idx=0)
        self.cnn = nn.Conv1d(dc, 64, 3, padding=1)
        self.lstm = nn.LSTM(dw + 64, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.drop = nn.Dropout(0.4); self.fc = nn.Linear(2 * h, len(TAGS))
    def forward(self, w, c):
        B, L, K = c.shape
        ce = self.cnn(self.ec(c.flatten(0, 1)).transpose(1, 2)).max(2).values.unflatten(0, (B, L))
        o, _ = self.lstm(self.drop(torch.cat([self.ew(w), ce], -1)))
        return self.fc(self.drop(o))


def train_tagger(mau, EP=25):
    m = Tagger(len(stoi) + 2, len(ctoi) + 2).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(EP):
        m.train(); random.shuffle(mau)
        for i in range(0, len(mau), 32):
            W, C, Y = lo(mau[i:i + 32], 0.1)
            loss = nn.functional.cross_entropy(m(W, C).flatten(0, 1), Y.flatten(), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def cum(tag, loai):
    out, bd = [], None
    for i, t in enumerate(tag + ["O"]):
        if t == "B-" + loai or (t == "I-" + loai and bd is None):
            if bd is not None: out.append((bd, i - 1))
            bd = i
        elif t != "I-" + loai and bd is not None:
            out.append((bd, i - 1)); bd = None
    return out


# category và VA học trên bộ tứ có aspect / opinion thật (bỏ NULL)
def chu(s):
    return " ".join(chuan(w) for w in re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", s))

def menh_de(text, sp, i):
    w = [text[a:b].lower() for a, b in sp]; l = r = i
    while l > 0 and w[l - 1] not in NGAT: l -= 1
    while r < len(w) - 1 and w[r + 1] not in NGAT: r += 1
    return " ".join(chuan(x) for x in w[l:r + 1])

q_tr = []
for r in train + dev:
    t, sp, _ = gan_nhan(r)
    for q in r["quadruplets"]:
        if q["aspect"] == "NULL" or q["opinion"] == "NULL": continue
        st = t.lower().find(q["opinion"].lower()); k = next((j for j, (a, b) in enumerate(sp) if a >= st), 0)
        q_tr.append((chu(q["aspect"]), chu(q["opinion"]), menh_de(t, sp, k), q["category"], q["valence"], q["arousal"]))
v1, v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True), TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
f_cat = lambda X, fit=False: hstack([v1.fit_transform(X), v2.fit_transform(X)]) if fit else hstack([v1.transform(X), v2.transform(X)])
clf = LogisticRegression(C=8, max_iter=4000).fit(f_cat([f"{a} | {o}" for a, o, *_ in q_tr], True), [x[3] for x in q_tr])
v3, v4 = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True), TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
f_va = lambda O, M, fit=False: hstack([v3.fit_transform(O), 0.5 * v4.fit_transform(M)]) if fit else hstack([v3.transform(O), 0.5 * v4.transform(M)])
Xva = f_va([x[1] for x in q_tr], [x[2] for x in q_tr], True)
reg = [Ridge(alpha=1.5).fit(Xva, [x[k] for x in q_tr]) for k in (4, 5)]


def du_doan(model, rows):
    kq = []
    with torch.inference_mode():
        for r in rows:
            text, sp = token(r["text"])
            if not sp: kq.append({"id": r["id"], "quadruplets": []}); continue
            W, C, _ = lo([(text, sp, None)])
            tag = [TAGS[i] for i in model(W, C)[0].argmax(-1).tolist()]
            md = np.cumsum([text[a:b].lower() in NGAT for a, b in sp])
            A, O = cum(tag, "ASP"), cum(tag, "OPI"); quads, da = [], set()
            for o0, o1 in O:
                cung = [x for x in A if md[x[0]] == md[o0]] or [x for x in A if abs(x[0] - o0) <= 8]
                if not cung: continue                                    # test không có NULL: bỏ
                a0, a1 = min(cung, key=lambda x: abs(x[0] - o0))
                asp, op = text[sp[a0][0]:sp[a1][1]], text[sp[o0][0]:sp[o1][1]]
                cat = clf.predict(f_cat([f"{chu(asp)} | {chu(op)}"]))[0]
                X = f_va([chu(op)], [menh_de(text, sp, o0)])
                v, a = (float(np.clip(m.predict(X)[0], 1, 9).round(2)) for m in reg)
                k = (asp.lower(), cat.lower(), op.lower())
                if k not in da: da.add(k); quads.append(dict(aspect=asp, category=cat, opinion=op, valence=v, arousal=a))
            kq.append({"id": r["id"], "quadruplets": quads})
    return kq


def cf1(gold, pred):
    ctp = tp = fp = fn = 0.0
    P = {x["id"]: x["quadruplets"] for x in pred}
    for r in gold:
        pr, khop = P.get(r["id"], []), 0
        kh = lambda q: (q["aspect"].lower(), q["category"].lower(), q["opinion"].lower())
        for g in r["quadruplets"]:
            d = [max(0, 1 - np.hypot(p["valence"] - g["valence"], p["arousal"] - g["arousal"]) / np.sqrt(128)) for p in pr if kh(p) == kh(g)]
            if len(d) == 1: ctp += d[0]; tp += 1; khop += 1
            else: fn += 1
        fp += len(pr) - khop
    p, rc = ctp / max(tp + fp, 1), ctp / max(tp + fn, 1)
    return 2 * p * rc / (p + rc) if p + rc else 0.0


m = train_tagger([gan_nhan(r) for r in train])                 # tự chấm: tagger chỉ train trên train
print(f"dev cF1 = {cf1(dev, du_doan(m, dev)):.4f} (category / VA đã thấy dev nên hơi lạc quan)")
seed_all(42); m = train_tagger(mau_tr)                         # bản cuối: train + dev
out = du_doan(m, test)
with open("pred_test.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
pd.DataFrame({"id": [o["id"] for o in out], "quadruplets": [json.dumps(o["quadruplets"], ensure_ascii=False) for o in out]}).to_csv("submission.csv", index=False)
print(f"đã ghi pred_test.jsonl và submission.csv: {len(out)} câu, {sum(len(o['quadruplets']) for o in out)} bộ tứ")
