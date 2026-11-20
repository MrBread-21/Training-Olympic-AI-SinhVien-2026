import os, random, time
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
import cv2, numpy as np

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8
ids = sorted(f[:-4] for f in os.listdir("data/b_train/images")); te_ids = sorted(f[:-4] for f in os.listdir("data/b_test"))
rgb = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
X = np.stack([rgb(f"data/b_train/images/{i}.png") for i in ids]); M = np.stack([cv2.imread(f"data/b_train/masks/{i}.png", 0) > 127 for i in ids])
Xte = np.stack([rgb(f"data/b_test/{i}.png") for i in te_ids])
MEAN, STD = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None], torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]
prep = lambda a: (torch.from_numpy(a).to(DEV).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD

def conv(cin, cout): return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self):
        super().__init__(); r = torchvision.models.resnet18(weights="DEFAULT")
        self.stem, self.pool, self.l1, self.l2, self.l3, self.l4 = nn.Sequential(r.conv1, r.bn1, r.relu), r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4
        self.d3, self.d2, self.d1, self.d0 = conv(512 + 256, 256), conv(256 + 128, 128), conv(128 + 64, 64), conv(64 + 64, 64)
        self.out = nn.Sequential(conv(64, 32), nn.Conv2d(32, 1, 1))
    def forward(self, x):
        x0 = self.stem(x); x1 = self.l1(self.pool(x0)); x2 = self.l2(x1); x3 = self.l3(x2); x4 = self.l4(x3)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear"), b], 1)
        d = self.d0(up(self.d1(up(self.d2(up(self.d3(up(x4, x3)), x2)), x1)), x0))
        return self.out(F.interpolate(d, size=x.shape[-2:], mode="bilinear"))[:, 0]

def dice(p, t, eps=1.0): return ((2 * (p * t).sum((1, 2)) + eps) / (p.sum((1, 2)) + t.sum((1, 2)) + eps))

@torch.no_grad()
def predict(m, X):
    m.eval()
    with torch.inference_mode(), torch.autocast(DEV, torch.bfloat16, enabled=AMP):
        return torch.cat([torch.sigmoid(m(prep(X[i:i + 64])).float()) for i in range(0, len(X), 64)])

perm = np.random.permutation(len(X)); iva, itr = perm[:40], perm[40:]
model, EP, BS = UNet().to(DEV), 30, 32
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * (len(itr) // BS), pct_start=0.1)
best, t0 = -1, time.time()
for ep in range(EP):
    model.train()
    for b in np.random.permutation(itr)[: len(itr) // BS * BS].reshape(-1, BS):     # chia index thành các batch
        x, t = prep(X[b]), torch.from_numpy(M[b]).float().to(DEV)
        if random.random() < 0.5: x, t = x.flip(3), t.flip(2)
        if random.random() < 0.5: x, t = x.flip(2), t.flip(1)
        with torch.autocast(DEV, torch.bfloat16, enabled=AMP):
            lo = model(x).float()
        loss = F.binary_cross_entropy_with_logits(lo, t) + (1 - dice(torch.sigmoid(lo), t)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    d = dice((predict(model, X[iva]) > 0.5).float(), torch.from_numpy(M[iva]).float().to(DEV)).mean().item()   # val ngoài vòng batch
    if d > best: best = d; torch.save(model.state_dict(), "best_unet.pt")
    print(f"ep {ep} loss {loss.item():.3f} val Dice {d:.4f} ({time.time() - t0:.0f}s)", flush=True)

model.load_state_dict(torch.load("best_unet.pt", map_location=DEV))
os.makedirs("masks_pred", exist_ok=True)
for i, p in zip(te_ids, predict(model, Xte).cpu().numpy()):
    cv2.imwrite(f"masks_pred/{i}.png", ((p > 0.5) * 255).astype(np.uint8))
print("best val Dice", round(best, 4), "| đã ghi masks_pred/", len(te_ids), "mặt nạ")
