import random, time
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torchvision.transforms import v2
import cv2, numpy as np, pandas as pd

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8
tr, sub = pd.read_csv("data/train.csv"), pd.read_csv("data/sample_submission.csv")
CLS = sorted(tr.label.unique()); K = len(CLS)
read = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
X = np.stack([read(f"data/train/{i}.png") for i in tr.id]); y = tr.label.map(CLS.index).values
Xte = np.stack([read(f"data/test/{i}.png") for i in sub.id])
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
# augment mô phỏng test: xoay/lật, RandAugment, mờ, JPEG, nhiễu
AUG = v2.Compose([v2.RandomHorizontalFlip(), v2.RandAugment(2, 9),
                  v2.RandomApply([v2.GaussianBlur(5, (0.3, 1.5))], 0.5), v2.RandomApply([v2.JPEG((10, 60))], 0.5)])

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y=None, train=False): self.X, self.y, self.train = X, y, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).permute(2, 0, 1)
        if self.train:
            x = AUG(torch.rot90(x, random.randint(0, 3), (1, 2))).float() / 255
            x = x + torch.randn_like(x) * random.uniform(0, 0.08)
        else: x = x.float() / 255
        x = (x - MEAN) / STD
        return (x, int(self.y[i])) if self.y is not None else x

MIX = v2.RandomChoice([v2.MixUp(num_classes=K), v2.CutMix(num_classes=K)])

def train(idx, epochs, lr=1e-3):
    m = torchvision.models.resnet18(weights="DEFAULT"); m.fc = nn.Linear(512, K); m = m.to(DEV)
    dl = torch.utils.data.DataLoader(DS(X[idx], y[idx], True), 64, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=0.2)
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = MIX(xb, yb)                                                   # nhãn mềm (B, K)
            with torch.autocast(DEV, torch.bfloat16, enabled=AMP):
                loss = F.cross_entropy(m(xb.to(DEV)).float(), yb.to(DEV), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        print(f"  ep {ep} loss {loss.item():.3f}", flush=True)
    return m

@torch.no_grad()
def predict(m, X, tta=True):
    m.eval(); out = []
    with torch.inference_mode(), torch.autocast(DEV, torch.bfloat16, enabled=AMP):
        for x in torch.utils.data.DataLoader(DS(X), 256, shuffle=False, num_workers=0):
            x, p = x.to(DEV), 0
            for k in (range(4) if tta else [0]):
                for f in ((False, True) if tta else (False,)):
                    v = torch.rot90(x, k, (2, 3)); v = v.flip(3) if f else v
                    p = p + m(v).float().softmax(1)
            out.append(p.cpu())
    return (torch.cat(out) / (8 if tta else 1)).numpy()

t0 = time.time()
m1 = train(np.arange(len(y)), 8)                                                  # giai đoạn 1: toàn bộ nhãn nhiễu
P = predict(m1, X, tta=False)
loss = -np.log(P[np.arange(len(y)), y] + 1e-8)
keep = np.concatenate([np.where(y == c)[0][np.argsort(loss[y == c])[:int(0.6 * (y == c).sum())]] for c in range(K)])
print(f"giữ {len(keep)}/{len(y)} mẫu loss nhỏ ({time.time() - t0:.0f}s)")
m2 = train(keep, 10)                                                               # giai đoạn 2: chỉ mẫu loss nhỏ
pred = (predict(m2, Xte) + 0.5 * predict(m1, Xte)).argmax(1)
sub["label"] = sub.id.map(dict(zip(sub.id, [CLS[k] for k in pred])))
assert sub.label.notna().all()
sub.to_csv("submission.csv", index=False)
print("xong", f"{time.time() - t0:.0f}s", sub.label.value_counts().to_dict())
