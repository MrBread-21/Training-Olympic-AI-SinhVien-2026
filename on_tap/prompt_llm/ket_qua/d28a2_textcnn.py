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
train = pd.read_csv("data/a_train.csv")
test = pd.read_csv("data/a_test.csv")
classes = sorted(train.label.unique())
vocab = {"<pad>": 0, "<unk>": 1}
cnt = Counter(w for t in train.text for w in re.findall(r"\w+", unicodedata.normalize("NFC", t).lower()))
for w, c in cnt.items():
    if c >= 2:
        vocab[w] = len(vocab)
def enc(s):
    return [vocab.get(w, 1) for w in re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())][:60] or [1]
X_tr = [enc(t) for t in train.text]
X_te = [enc(t) for t in test.text]
y = torch.tensor([classes.index(l) for l in train.label])
tr_idx, va_idx = train_test_split(np.arange(len(X_tr)), test_size=0.1, stratify=y.numpy(), random_state=42)

def batches(X, idx, ys=None, shuffle=False):
    order = sorted(idx, key=lambda i: len(X[i]))
    chunks = [order[k:k+64] for k in range(0, len(order), 64)]
    if shuffle:
        random.shuffle(chunks)
    for pos in chunks:
        xb = pad_sequence([torch.tensor(X[i]) for i in pos], batch_first=True, padding_value=0)
        if xb.size(1) < 4:
            xb = F.pad(xb, (0, 4 - xb.size(1)))
        yb = ys[pos].to(dev) if ys is not None else None
        yield pos, xb.to(dev), yb

class TextCNN(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, 128, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(128, 100, k) for k in (2, 3, 4)])
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(300, 5)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([F.relu(c(e)).max(dim=2).values for c in self.convs], 1)
        return self.fc(self.drop(h))
model = TextCNN(len(vocab)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
def predict(X, idx):
    model.eval()
    out = np.zeros(len(X), dtype=int)
    with torch.inference_mode(), amp():
        for pos, xb, _ in batches(X, idx, None, False):
            out[pos] = model(xb).argmax(1).cpu().numpy()
    return out[idx]
best_f1 = 0
for epoch in range(12):
    model.train()
    for pos, xb, yb in batches(X_tr, tr_idx, y, True):
        with amp():
            logits = model(xb)
            loss = crit(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    va_pred = predict(X_tr, va_idx)
    va_true = y[va_idx].numpy()
    f1 = f1_score(va_true, va_pred, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_a2.pt")
    print(f"Epoch {epoch+1}, Val F1: {f1:.4f}")
model.load_state_dict(torch.load("best_a2.pt"))
test_pred = predict(X_te, np.arange(len(X_te)))
submission = pd.DataFrame({
    "id": test.id,
    "label": [classes[i] for i in test_pred]
})
submission.to_csv("submission_a2.csv", index=False)
