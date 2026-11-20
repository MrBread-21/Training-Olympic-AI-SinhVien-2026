import math, random, time, unicodedata
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = lambda s: unicodedata.normalize("NFC", str(s)).lower().split()
tr, va, te = pd.read_csv("data/train.csv"), pd.read_csv("data/val.csv"), pd.read_csv("data/test.csv")
vocab = lambda texts: ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted({w for s in texts for w in tok(s)})
SV, TV = vocab(tr.en), vocab(tr.vi); s2i, t2i = {w: i for i, w in enumerate(SV)}, {w: i for i, w in enumerate(TV)}
enc_s = lambda s: [s2i.get(w, 1) for w in tok(s)]
enc_t = lambda s: [2] + [t2i.get(w, 1) for w in tok(s)] + [3]
pad = lambda seqs: torch.nn.utils.rnn.pad_sequence([torch.tensor(s) for s in seqs], batch_first=True).to(DEV)

class Seq2Seq(nn.Module):
    def __init__(self, d=256, h=4, L=3, maxlen=40):
        super().__init__(); self.d = d
        self.se, self.te, self.pos = nn.Embedding(len(SV), d, padding_idx=0), nn.Embedding(len(TV), d, padding_idx=0), nn.Embedding(maxlen, d)
        self.tf = nn.Transformer(d, h, L, L, 4 * d, 0.1, batch_first=True); self.out = nn.Linear(d, len(TV))
    def emb(self, E, x): return E(x) * math.sqrt(self.d) + self.pos(torch.arange(x.shape[1], device=x.device))
    def encode(self, s): return self.tf.encoder(self.emb(self.se, s), src_key_padding_mask=s == 0)
    def decode(self, mem, s, t):
        mask = self.tf.generate_square_subsequent_mask(t.shape[1], device=t.device)
        return self.out(self.tf.decoder(self.emb(self.te, t), mem, tgt_mask=mask, tgt_key_padding_mask=t == 0, memory_key_padding_mask=s == 0))
@torch.no_grad()
def greedy(m, srcs, maxlen=30):
    m.eval(); out = []
    with torch.inference_mode():
        for i in range(0, len(srcs), 256):
            s = pad([enc_s(x) for x in srcs[i:i + 256]]); mem = m.encode(s); ys = torch.full((len(s), 1), 2, device=DEV)
            for _ in range(maxlen):
                ys = torch.cat([ys, m.decode(mem, s, ys)[:, -1].argmax(1, keepdim=True)], 1)
                if (ys == 3).any(1).all(): break
            out += [" ".join(TV[k] for k in row[1:].tolist()[:(row[1:].tolist() + [3]).index(3)]) for row in ys]
    return out
@torch.no_grad()
def beam(m, src, k=4, maxlen=30, alpha=0.7):
    m.eval(); s = pad([enc_s(src)]); mem = m.encode(s); hyps = [([2], 0.0)]
    with torch.inference_mode():
        for _ in range(maxlen):
            cand = []
            for y, sc in hyps:
                if y[-1] == 3: cand.append((y, sc)); continue
                lp = m.decode(mem, s, torch.tensor([y], device=DEV))[0, -1].log_softmax(-1)
                cand += [(y + [j], sc + v) for v, j in zip(*map(torch.Tensor.tolist, lp.topk(k)))]
            hyps = sorted(cand, key=lambda c: c[1] / len(c[0]) ** alpha, reverse=True)[:k]   # length penalty
            if all(y[-1] == 3 for y, _ in hyps): break
    return " ".join(TV[j] for j in hyps[0][0][1:] if j > 3)
def bleu(hyp, ref, N=4):
    p = []
    for n in range(1, N + 1):
        g = lambda t: Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
        hit = sum(sum((g(h.split()) & g(r.split())).values()) for h, r in zip(hyp, ref)); tot = sum(max(len(h.split()) - n + 1, 0) for h in hyp)
        p.append(math.log(max(hit, 1e-9) / max(tot, 1)))
    lh, lr = sum(len(h.split()) for h in hyp), sum(len(r.split()) for r in ref)
    return math.exp(min(0, 1 - lr / max(lh, 1)) + sum(p) / N)

S, T = [enc_s(x) for x in tr.en], [enc_t(x) for x in tr.vi]
m, EP, nb = Seq2Seq().to(DEV), 10, len(S) // 128
opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=EP * nb, pct_start=0.1)
best, t0 = -1, time.time()
for ep in range(EP):
    m.train(); order = np.argsort([len(x) + random.random() for x in S])        # bucketing theo độ dài câu nguồn
    for b in random.sample(np.array_split(order, nb), nb):
        s, t = pad([S[i] for i in b]), pad([T[i] for i in b])
        lo = m.decode(m.encode(s), s, t[:, :-1])
        loss = F.cross_entropy(lo.transpose(1, 2), t[:, 1:], ignore_index=0, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    sc = bleu(greedy(m, list(va.en)), list(va.vi))                                # val ngoài vòng batch
    if sc > best: best = sc; torch.save(m.state_dict(), "best_seq2seq.pt")
    print(f"ep {ep} loss {loss.item():.3f} val BLEU {sc:.4f} ({time.time() - t0:.0f}s)", flush=True)
m.load_state_dict(torch.load("best_seq2seq.pt", map_location=DEV))
print("val 100 câu: greedy BLEU %.4f | beam BLEU %.4f" % (bleu(greedy(m, list(va.en[:100])), list(va.vi[:100])),
                                                      bleu([beam(m, x) for x in va.en[:100]], list(va.vi[:100]))))
pd.DataFrame({"id": te.id, "vi": greedy(m, list(te.en))}).to_csv("submission.csv", index=False)
print("đã ghi submission.csv, best val BLEU", round(best, 4))
