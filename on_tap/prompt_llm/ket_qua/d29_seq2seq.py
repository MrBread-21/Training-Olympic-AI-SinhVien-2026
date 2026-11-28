import torch
import os, random, json, math, re, time, unicodedata, heapq
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.io import wavfile
from scipy.sparse import hstack
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, StratifiedGroupKFold,
                                     GroupShuffleSplit, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


seed_all(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def amp():
    return torch.autocast(dev, torch.bfloat16, enabled=use_bf16)
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

src_words = set()
tgt_words = set()
for s in train['en']:
    src_words.update(s.split())
for s in train['vi']:
    tgt_words.update(s.split())

sv = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
for w in src_words:
    if w not in sv:
        sv[w] = len(sv)
tv = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
for w in tgt_words:
    if w not in tv:
        tv[w] = len(tv)

t_itos = {v:k for k,v in tv.items()}

def enc_src(s):
    return [sv.get(w,1) for w in s.split()]
def enc_tgt(s):
    return [2] + [tv.get(w,1) for w in s.split()] + [3]

def make_batches(srcs, tgts, bs, shuffle):
    idx = list(range(len(srcs)))
    idx.sort(key=lambda x: len(srcs[x]))
    batches = []
    for i in range(0, len(idx), bs):
        batch = idx[i:i+bs]
        if shuffle:
            random.shuffle(batch)
        src = [srcs[i] for i in batch]
        tgt = [tgts[i] for i in batch] if tgts is not None else None
        src = pad_sequence([torch.LongTensor(s) for s in src], batch_first=True, padding_value=0)
        if tgt is not None:
            tgt = pad_sequence([torch.LongTensor(t) for t in tgt], batch_first=True, padding_value=0)
        src = src.to(dev)
        tgt = tgt.to(dev) if tgt is not None else None
        batches.append((batch, src, tgt))
    return batches

class Model(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, 256, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, 256, padding_idx=0)
        self.positional = nn.Embedding(64, 256)
        self.transformer = nn.Transformer(256, 4, 3, 3, 512, 0.1, batch_first=True)
        self.fc = nn.Linear(256, tgt_vocab_size)
    
    def forward(self, src, tgt_in):
        src_pad = src == 0  # SỬA TAY: mặt nạ phải tính từ id trước khi embedding
        src = self.src_embedding(src) * 16
        pos = torch.arange(src.size(1), device=dev).unsqueeze(0).repeat(src.size(0), 1)
        src = src + self.positional(pos)
        tgt = self.tgt_embedding(tgt_in) * 16
        pos = torch.arange(tgt.size(1), device=dev).unsqueeze(0).repeat(tgt.size(0), 1)
        tgt = tgt + self.positional(pos)
        src_key_padding_mask = src_pad
        tgt_key_padding_mask = tgt_in == 0
        memory_key_padding_mask = src_pad
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=dev)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        out = self.fc(out)
        return out
def bleu(hyps, refs):
    from math import exp, log
    from collections import Counter
    num = [0] * 4
    den = [0] * 4
    c = r = 0
    for h, ref in zip(hyps, refs):
        c += len(h)
        r += len(ref)
        for n in range(1, 5):
            hgrams = Counter([tuple(h[i:i+n]) for i in range(len(h)-n+1)])
            rgrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)])
            num[n-1] += sum(min(hgrams[g], rgrams.get(g, 0)) for g in hgrams)
            den[n-1] += max(len(h) - n + 1, 0)
    if min(num) == 0:
        return 0.0
    bp = 1.0 if c > r else exp(1 - r / c)
    return bp * exp(sum(log(num[k]/den[k]) for k in range(4)) / 4)

assert abs(bleu([["a", "b", "c", "d", "e"]], [["a", "b", "c", "d", "e"]]) - 1) < 1e-12
assert bleu([["a", "b", "c", "d"]], [["e", "f", "g", "h"]]) == 0
assert 0 < bleu([["a", "b", "c", "d", "x"]], [["a", "b", "c", "d", "e"]]) < 1
def greedy(model, src, max_len=40):
    B = src.size(0)
    ys = torch.full((B, 1), 2, dtype=torch.long, device=src.device)
    for _ in range(max_len):
        with amp():
            logits = model(src, ys)[:, -1]
        next_words = logits.argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_words], 1)
        if (ys == 3).all():
            break
    hyps = []
    for y in ys:
        y = y.tolist()
        if 3 in y:
            y = y[:y.index(3)+1]
        else:
            y.append(3)
        hyps.append([t_itos[i] for i in y[1:-1]])
    return hyps

tr_src = [enc_src(s) for s in train.en]
tr_tgt = [enc_tgt(s) for s in train.vi]
val_src = [enc_src(s) for s in val.en]
val_refs = [s.split() for s in val.vi]

model = Model(len(sv), len(tv)).to(dev)
optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98))

def translate(srcs):
    model.eval()
    out = [None] * len(srcs)
    with torch.inference_mode(), amp():
        for idx, src, _ in make_batches(srcs, None, 128, False):
            hyps = greedy(model, src)
            for i, h in zip(idx, hyps):
                out[i] = h
    return out

best_bleu = 0
for epoch in range(10):
    model.train()
    total_loss = 0
    batches = make_batches(tr_src, tr_tgt, 64, True)
    for _, src, tgt in batches:
        tgt_in = tgt[:, :-1]
        gold = tgt[:, 1:]
        optim.zero_grad()
        with amp():
            logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.float().transpose(1, 2), gold, ignore_index=0, label_smoothing=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(batches)
    val_hyps = translate(val_src)
    val_bleu = bleu(val_hyps, val_refs)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val BLEU: {val_bleu:.4f}")
    if val_bleu > best_bleu:
        best_bleu = val_bleu
        torch.save(model.state_dict(), "best.pt")

model.load_state_dict(torch.load("best.pt"))
test_src = [enc_src(s) for s in test.en]
test_hyps = translate(test_src)
submission = pd.DataFrame({
    "id": test.id,
    "vi": [" ".join(h) for h in test_hyps]
})
submission.to_csv("submission.csv", index=False)
assert len(submission) == 1000
