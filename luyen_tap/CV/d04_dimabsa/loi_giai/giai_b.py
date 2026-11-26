import torch                                                   # Windows: nạp torch trước pandas / sklearn
import json, random, re, unicodedata
import numpy as np
from torch import nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEEN = {"ko": "không", "k": "không", "qá": "quá", "wá": "quá", "dc": "được", "đc": "được"}
NGAT = {",", ".", "!", ":", "nhưng", "mà", "và", "còn"}
TAGS = ["O", "B-ASP", "I-ASP", "B-OPI", "I-OPI"]
chuan = lambda w: TEEN.get(w.lower(), w.lower())

def token(text):
    text = unicodedata.normalize("NFC", text)
    return text, [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]

def gan_nhan(r):
    """Nhãn BIO từ bộ tứ: tìm vị trí ký tự của aspect / opinion trùng ranh giới token (bỏ qua token đã gán)."""
    text, sp = token(r["text"]); tag = ["O"] * len(sp)
    for q in r["quadruplets"]:
        for khoa, loai in (("aspect", "ASP"), ("opinion", "OPI")):
            s = q[khoa]
            if s == "NULL": continue
            st = -1
            while True:
                st = text.find(s, st + 1)
                if st < 0: break
                idx = [k for k, (a, b) in enumerate(sp) if a >= st and b <= st + len(s)]
                if idx and sp[idx[0]][0] == st and sp[idx[-1]][1] == st + len(s) and all(tag[k] == "O" for k in idx):
                    tag[idx[0]] = "B-" + loai
                    for k in idx[1:]: tag[k] = "I-" + loai
                    break
    return text, sp, tag

doc = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
tr, te = doc("data/b_train.jsonl"), doc("data/b_test.jsonl")
mau = [gan_nhan(r) for r in tr]
tu = sorted({chuan(t[a:b]) for t, sp, _ in mau for a, b in sp}); stoi = {w: i + 2 for i, w in enumerate(tu)}       # 0 pad, 1 unk
ky = sorted({c for t, _, _ in mau for c in t.lower()}); ctoi = {c: i + 2 for i, c in enumerate(ky)}

def ma_hoa(text, sp, word_drop=0.0):
    w = [1 if random.random() < word_drop else stoi.get(chuan(text[a:b]), 1) for a, b in sp]
    c = [[ctoi.get(ch, 1) for ch in text[a:b].lower()[:12]] for a, b in sp]
    return w, c

def lo(batch, word_drop=0.0):                                    # batch: list (text, sp, tag)
    L = max(len(sp) for _, sp, _ in batch); K = 12
    W = torch.zeros(len(batch), L, dtype=torch.long); C = torch.zeros(len(batch), L, K, dtype=torch.long)
    Y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (t, sp, tag) in enumerate(batch):
        w, c = ma_hoa(t, sp, word_drop); W[i, :len(w)] = torch.tensor(w)
        for j, cc in enumerate(c): C[i, j, :len(cc)] = torch.tensor(cc)
        if tag: Y[i, :len(tag)] = torch.tensor([TAGS.index(x) for x in tag])
    return W.to(DEV), C.to(DEV), Y.to(DEV)

class Tagger(nn.Module):
    def __init__(self, Vw, Vc, T=5, dw=100, dc=32, h=128):
        super().__init__()
        self.ew = nn.Embedding(Vw, dw, padding_idx=0); self.ec = nn.Embedding(Vc, dc, padding_idx=0)
        self.cnn = nn.Conv1d(dc, 64, 3, padding=1)
        self.lstm = nn.LSTM(dw + 64, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(2 * h, T)
    def forward(self, w, c):
        B, L, K = c.shape
        ce = self.cnn(self.ec(c.flatten(0, 1)).transpose(1, 2)).max(2).values.unflatten(0, (B, L))   # đặc trưng ký tự mỗi từ
        o, _ = self.lstm(self.drop(torch.cat([self.ew(w), ce], -1)))
        return self.fc(self.drop(o))

model = Tagger(len(stoi) + 2, len(ctoi) + 2).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
for ep in range(12):
    model.train(); random.shuffle(mau)
    for i in range(0, len(mau), 32):
        W, C, Y = lo(mau[i:i + 32], word_drop=0.1)                # word dropout: tập xử lý từ chưa gặp (aspect mới ở test)
        loss = nn.functional.cross_entropy(model(W, C).flatten(0, 1), Y.flatten(), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()

# category và VA học trên bộ tứ train
q_tr = [q for r in tr for q in r["quadruplets"]]
chu = lambda s: " ".join(chuan(w) for w in re.findall(r"\w+", unicodedata.normalize("NFC", s)))
xc = [f"{chu(q['aspect'])} | {chu(q['opinion'])}" for q in q_tr]
cv1, cv2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), TfidfVectorizer(ngram_range=(1, 2))
f_cat = lambda X, fit=False: hstack([cv1.fit_transform(X), cv2.fit_transform(X)]) if fit else hstack([cv1.transform(X), cv2.transform(X)])
clf = LogisticRegression(C=10, max_iter=3000).fit(f_cat(xc, True), [q["category"] for q in q_tr])
xo = [chu(q["opinion"]) for q in q_tr]
vo1, vo2 = TfidfVectorizer(ngram_range=(1, 3)), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
f_va = lambda X, fit=False: hstack([vo1.fit_transform(X), vo2.fit_transform(X)]) if fit else hstack([vo1.transform(X), vo2.transform(X)])
Xo = f_va(xo, True); reg = [Ridge(alpha=0.1).fit(Xo, [q[k] for q in q_tr]) for k in ("valence", "arousal")]

def cum(tag, loai):                                               # các cụm (đầu, cuối) theo BIO, sửa I- đứng sau O thành B-
    out, bd = [], None
    for i, t in enumerate(tag + ["O"]):
        if t == "B-" + loai or (t == "I-" + loai and bd is None):
            if bd is not None: out.append((bd, i - 1))
            bd = i
        elif t != "I-" + loai and bd is not None:
            out.append((bd, i - 1)); bd = None
    return out

model.eval(); kq = []
with torch.inference_mode():
    for r in te:
        text, sp = token(r["text"]); W, C, _ = lo([(text, sp, None)])
        tag = [TAGS[i] for i in model(W, C)[0].argmax(-1).tolist()]
        menh_de = np.cumsum([text[a:b].lower() in NGAT for a, b in sp])          # chỉ số mệnh đề của từng token
        A, O = cum(tag, "ASP"), cum(tag, "OPI")
        quads, da = [], set()
        for o0, o1 in O:
            cung = [x for x in A if menh_de[x[0]] == menh_de[o0]]
            asp = text[sp[min(cung, key=lambda x: abs(x[0] - o0))[0]][0]:sp[min(cung, key=lambda x: abs(x[0] - o0))[1]][1]] if cung else "NULL"
            op = text[sp[o0][0]:sp[o1][1]]
            cat = clf.predict(f_cat([f"{chu(asp)} | {chu(op)}"]))[0]
            Xq = f_va([chu(op)]); v, a = (float(np.clip(m.predict(Xq)[0], 1, 9).round(2)) for m in reg)
            k = (asp.lower(), cat.lower(), op.lower())
            if k not in da:                                                       # khử trùng lặp
                da.add(k); quads.append(dict(aspect=asp, category=cat, opinion=op, valence=v, arousal=a))
        kq.append({"id": r["id"], "quadruplets": quads})
with open("pred_b.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in kq)
print("đã ghi pred_b.jsonl")
