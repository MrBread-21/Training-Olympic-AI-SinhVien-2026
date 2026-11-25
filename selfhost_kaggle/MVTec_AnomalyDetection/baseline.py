"""Baseline phát hiện bất thường: đặc trưng ResNet-18 toàn ảnh, điểm = khoảng cách tới ảnh bình thường gần nhất cùng loại.
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
from PIL import Image

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


tr, te = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
Ftr = dac_trung([D / f"train/{i}.png" for i in tr.id]); Fte = dac_trung([D / f"test/{i}.png" for i in te.id])
score = np.zeros(len(te))
for c in te.category.unique():
    a, b = Ftr[tr.category.values == c], Fte[te.category.values == c]
    d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    score[te.category.values == c] = np.sqrt(d.min(1))
pd.DataFrame({"id": te.id, "score": score}).to_csv("submission.csv", index=False)
print("xong")
