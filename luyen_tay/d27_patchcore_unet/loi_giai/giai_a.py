import os, random
import torch, torch.nn.functional as F, torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import cv2, numpy as np, pandas as pd

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8

def read(folder):
    ids = sorted(f[:-4] for f in os.listdir(folder))
    return ids, np.stack([cv2.cvtColor(cv2.imread(f"{folder}/{i}.png"), cv2.COLOR_BGR2RGB) for i in ids])

_, Xtr = read("data/a_train"); te_ids, Xte = read("data/a_test")
net = torchvision.models.resnet18(weights="DEFAULT").to(DEV).eval()
ext = create_feature_extractor(net, {"layer2": "l2", "layer3": "l3"})
MEAN, STD = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None], torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]

@torch.no_grad()
def patches(X, bs=64):
    """Đặc trưng patch (N, 256 vị trí, 384 kênh): layer2 + layer3 phóng về 16x16, làm mượt 3x3."""
    out = []
    with torch.inference_mode(), torch.autocast(DEV, torch.bfloat16, enabled=AMP):
        for i in range(0, len(X), bs):
            x = F.interpolate(torch.from_numpy(X[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float() / 255, size=128, mode="bilinear")
            o = ext((x - MEAN) / STD)
            f = torch.cat([o["l2"], F.interpolate(o["l3"], size=o["l2"].shape[-2:], mode="bilinear")], 1).float()
            f = F.avg_pool2d(f, 3, 1, 1)
            out.append(f.permute(0, 2, 3, 1).flatten(1, 2))                  # (B, C, H, W) sang (B, H*W, C) bằng permute
    return torch.cat(out)

bank = patches(Xtr).flatten(0, 1)
bank = bank[torch.randperm(len(bank), device=bank.device)[:16000]]          # memory bank rút gọn (coreset ngẫu nhiên)
P = patches(Xte)
score = torch.stack([torch.cdist(p, bank).min(1).values.max() for p in P]).cpu().numpy()   # điểm ảnh = patch xa nhất
pd.DataFrame({"id": te_ids, "score": score}).to_csv("submission_a.csv", index=False)
print("đã ghi submission_a.csv", len(te_ids), "ảnh; memory bank", tuple(bank.shape))
