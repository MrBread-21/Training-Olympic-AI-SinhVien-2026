"""Baseline đa nhãn: đặc trưng ResNet-18 ImageNet + 20 LogisticRegression (một lớp một mô hình), ngưỡng 0.5.
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

torch.manual_seed(42); np.random.seed(42)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
m = torchvision.models.resnet18(weights="IMAGENET1K_V1"); m.fc = torch.nn.Identity(); m = m.eval().to(DEV)
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def dac_trung(paths, bs=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), bs):
            out.append(m(torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEV)).cpu().numpy())
    return np.concatenate(out)


tr, te = pd.read_csv(D / "train.csv", keep_default_na=False), pd.read_csv(D / "test.csv")
lop = pd.read_csv(D / "classes.csv").label.tolist()
Xtr = dac_trung([D / f"train/{i}.jpg" for i in tr.id]); Xte = dac_trung([D / f"test/{i}.jpg" for i in te.id])
P = np.zeros((len(te), len(lop)), dtype=bool)
for k, c in enumerate(lop):
    y = tr.labels.str.split().map(lambda s: c in s).astype(int)
    P[:, k] = LogisticRegression(max_iter=2000).fit(Xtr, y).predict_proba(Xte)[:, 1] > 0.5
pd.DataFrame({"id": te.id, "labels": [" ".join(c for c, b in zip(lop, r) if b) for r in P]}).to_csv("submission.csv", index=False)
print("xong")
