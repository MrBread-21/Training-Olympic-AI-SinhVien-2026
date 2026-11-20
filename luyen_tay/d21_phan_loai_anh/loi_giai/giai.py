import os, random, time
import torch, torch.nn as nn, torch.nn.functional as F, torchvision   # import torch trước cv2
import cv2, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8           # bf16 chỉ trên Ampere trở lên
D = "data"
tr, sub = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/sample_submission.csv")
CLS = sorted(tr.label.unique()); c2i = {c: i for i, c in enumerate(CLS)}
read = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
X = np.stack([read(f"{D}/train/{i}.png") for i in tr.id]); y = tr.label.map(c2i).values
Xte = np.stack([read(f"{D}/test/{i}.png") for i in sub.id])
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y=None, train=False): self.X, self.y, self.train = X, y, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).permute(2, 0, 1).float() / 255             # HWC sang CHW bằng permute
        if self.train:
            if random.random() < 0.5: x = x.flip(2)
            x = torch.rot90(x, random.randint(0, 3), (1, 2))
        x = (x - MEAN) / STD
        return (x, torch.tensor(self.y[i], dtype=torch.long)) if self.y is not None else x

def make_model():
    m = torchvision.models.resnet18(weights="DEFAULT")
    m.fc = nn.Linear(m.fc.in_features, len(CLS))
    return m.to(DEV)

@torch.no_grad()
def predict(m, X, bs=256):
    m.eval(); out = []
    with torch.inference_mode(), torch.autocast(DEV, torch.bfloat16, enabled=AMP):
        for x in torch.utils.data.DataLoader(DS(X), bs, shuffle=False, num_workers=0):
            out.append(m(x.to(DEV)).float().softmax(1).cpu())
    return torch.cat(out).numpy()

itr, iva = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
dl = torch.utils.data.DataLoader(DS(X[itr], y[itr], train=True), 64, shuffle=True, num_workers=0, drop_last=True)
model, EP = make_model(), 6
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)          # tạo một lần, ngoài vòng epoch
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * len(dl), pct_start=0.2)
best, t0 = -1, time.time()
for ep in range(EP):
    model.train()
    for xb, yb in dl:
        with torch.autocast(DEV, torch.bfloat16, enabled=AMP):
            loss = F.cross_entropy(model(xb.to(DEV)), yb.to(DEV), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    f1 = f1_score(y[iva], predict(model, X[iva]).argmax(1), average="macro")      # validation ngoài vòng batch
    if f1 > best: best = f1; torch.save(model.state_dict(), "best.pt")
    print(f"ep {ep} loss {loss.item():.3f} val F1 {f1:.4f} ({time.time() - t0:.0f}s)", flush=True)

model.load_state_dict(torch.load("best.pt", map_location=DEV))
pred = predict(model, Xte).argmax(1)
lab = dict(zip(sub.id, [CLS[k] for k in pred]))
sub["label"] = sub.id.map(lab)                                                    # điền theo id
assert sub.label.notna().all() and len(sub) == len(Xte)
sub.to_csv("submission.csv", index=False)
print("best val F1", round(best, 4), "| phân bố:", sub.label.value_counts().to_dict())
