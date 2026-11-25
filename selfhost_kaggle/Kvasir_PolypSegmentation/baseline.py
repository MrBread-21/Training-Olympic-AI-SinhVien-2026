"""Baseline polyp: U-Net nhỏ train từ đầu 8 epoch, ngưỡng 0.5, mã hoá RLE.
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

torch.manual_seed(42); np.random.seed(42)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rle_encode(mask):
    """mask (H, W) 0/1 -> 'start length ...', start từ 1, duyệt theo hàng (row-major)."""
    p = np.concatenate([[0], mask.flatten().astype(np.uint8), [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return " ".join(map(str, r))


def doc(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)


def khoi(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class UNet(nn.Module):
    def __init__(s, c=16):
        super().__init__()
        s.d1, s.d2, s.d3, s.b = khoi(3, c), khoi(c, 2 * c), khoi(2 * c, 4 * c), khoi(4 * c, 8 * c)
        s.u3, s.u2, s.u1 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), nn.ConvTranspose2d(2 * c, c, 2, 2)
        s.c3, s.c2, s.c1, s.o = khoi(8 * c, 4 * c), khoi(4 * c, 2 * c), khoi(2 * c, c), nn.Conv2d(c, 1, 1)

    def forward(s, x):
        a = s.d1(x); b = s.d2(nn.functional.max_pool2d(a, 2)); c = s.d3(nn.functional.max_pool2d(b, 2)); d = s.b(nn.functional.max_pool2d(c, 2))
        c = s.c3(torch.cat([s.u3(d), c], 1)); b = s.c2(torch.cat([s.u2(c), b], 1)); a = s.c1(torch.cat([s.u1(b), a], 1))
        return s.o(a)


tr, te = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
X = torch.stack([doc(D / f"train/images/{i}.jpg") for i in tr.id])
Y = torch.stack([torch.from_numpy(np.asarray(Image.open(D / f"train/masks/{i}.png")) > 127).float()[None] for i in tr.id])
net = UNet().to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)
for ep in range(8):
    net.train()
    for k in torch.randperm(len(X)).split(16):
        x, y = X[k].to(DEV), Y[k].to(DEV)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("epoch", ep, round(loss.item(), 4))
net.eval(); rles = []
with torch.inference_mode():
    for i in te.id:
        p = torch.sigmoid(net(doc(D / f"test/images/{i}.jpg")[None].to(DEV)))[0, 0].cpu().numpy()
        rles.append(rle_encode(p > 0.5))
pd.DataFrame({"id": te.id, "rle": rles}).to_csv("submission.csv", index=False)
print("xong")
