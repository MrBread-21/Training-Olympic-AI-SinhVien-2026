import random, re, time, unicodedata
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = lambda s: re.findall(r"\w+", unicodedata.normalize("NFC", str(s)).lower())
tr, te = pd.read_csv("data/a_train.csv"), pd.read_csv("data/a_test.csv")
CLS = sorted(tr.label.unique()); y = tr.label.map(CLS.index).values
cnt = Counter(w for s in tr.text for w in tok(s))
itos = ["<pad>", "<unk>"] + [w for w, c in cnt.items() if c >= 2]; stoi = {w: i for i, w in enumerate(itos)}
enc = lambda s: [stoi.get(w, 1) for w in tok(s)] or [1]
Str, Ste = [enc(s) for s in tr.text], [enc(s) for s in te.text]

def batches(seqs, bs, shuffle):
    """Bucketing: gom câu dài gần bằng nhau vào một batch để ít padding."""
    order = np.argsort([len(s) + (random.random() * 2 if shuffle else 0) for s in seqs])
    bl = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle: random.shuffle(bl)
    for b in bl:
        x = torch.zeros(len(b), max(len(seqs[i]) for i in b), dtype=torch.long)
        for j, i in enumerate(b): x[j, :len(seqs[i])] = torch.tensor(seqs[i])
        yield b, x.to(DEV)

class TextCNN(nn.Module):
    def __init__(self, V, K, d=128, nf=128):
        super().__init__(); self.emb = nn.Embedding(V, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, nf, k, padding=k // 2) for k in (2, 3, 4)])
        self.drop, self.fc = nn.Dropout(0.5), nn.Linear(3 * nf, K)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                                 # (B, L, d) sang (B, d, L)
        return self.fc(self.drop(torch.cat([F.relu(c(e)).max(2).values for c in self.convs], 1)))

@torch.no_grad()
def predict(m, seqs):
    m.eval(); P = np.zeros((len(seqs), len(CLS)))
    with torch.inference_mode():
        for b, x in batches(seqs, 256, False): P[b] = m(x).softmax(1).cpu().numpy()
    return P

perm = np.random.permutation(len(y)); iva, itr = perm[:300], perm[300:]
m, EP = TextCNN(len(itos), len(CLS)).to(DEV), 12
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
best, t0 = -1, time.time()
for ep in range(EP):
    m.train()
    for b, x in batches([Str[i] for i in itr], 64, True):
        loss = F.cross_entropy(m(x), torch.tensor(y[itr][b], device=DEV), label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    f1 = f1_score(y[iva], predict(m, [Str[i] for i in iva]).argmax(1), average="macro")   # val ngoài vòng batch
    if f1 > best: best = f1; torch.save(m.state_dict(), "best_textcnn.pt")
    print(f"ep {ep} loss {loss.item():.3f} val F1 {f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
m.load_state_dict(torch.load("best_textcnn.pt", map_location=DEV))
pd.DataFrame({"id": te.id, "label": [CLS[k] for k in predict(m, Ste).argmax(1)]}).to_csv("submission_a2.csv", index=False)
print("best val F1", round(best, 4), "| đã ghi submission_a2.csv")
