import random, time
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torchvision.transforms import v2
import cv2, numpy as np, pandas as pd

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8
read = lambda d, ids: np.stack([cv2.cvtColor(cv2.imread(f"data/{d}/{i}.png"), cv2.COLOR_BGR2RGB) for i in ids])
tr, sub = pd.read_csv("data/train.csv"), pd.read_csv("data/sample_submission.csv")
g_ids = sorted(p[:-4] for p in __import__("os").listdir("data/gallery"))
X, obj = read("train", tr.id), tr.object.astype("category").cat.codes.values
Xg, Xq = read("gallery", g_ids), read("query", sub.id)
va_obj = np.arange(80, 100); itr = np.where(obj < 80)[0]; iva = np.where(obj >= 80)[0]   # val = 20 đối tượng chưa thấy
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
AUG = v2.Compose([v2.RandomResizedCrop(64, (0.7, 1.0)), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.ColorJitter(0.3, 0.3, 0.3, 0.05)])
to_x = lambda a: (F.interpolate(a.float() / 255, size=128, mode="bilinear") - MEAN) / STD

class GeM(nn.Module):
    def __init__(self, p=3.0): super().__init__(); self.p = nn.Parameter(torch.tensor(p))
    def forward(self, x): return x.clamp(min=1e-6).pow(self.p).mean((2, 3)).pow(1 / self.p)

class Net(nn.Module):
    def __init__(self, n_cls, d=256, s=30.0, m=0.3):
        super().__init__(); r = torchvision.models.resnet18(weights="DEFAULT")
        self.body = nn.Sequential(*list(r.children())[:-2]); self.pool = GeM()
        self.emb = nn.Sequential(nn.Linear(512, d), nn.BatchNorm1d(d))
        self.W = nn.Parameter(torch.randn(n_cls, d) * 0.01); self.s, self.m = s, m
    def forward(self, x): return F.normalize(self.emb(self.pool(self.body(x))), dim=1)
    def arcface(self, e, y):                                                   # logit = s*cos(theta + m) ở lớp đúng
        cos = e @ F.normalize(self.W, dim=1).T
        th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        return self.s * torch.where(F.one_hot(y, cos.shape[1]).bool(), torch.cos(th + self.m), cos)

@torch.no_grad()
def embed(net, X, bs=128):
    net.eval(); out = []
    with torch.inference_mode(), torch.autocast(DEV, torch.bfloat16, enabled=AMP):
        for i in range(0, len(X), bs):
            x = to_x(torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2)).to(DEV)
            out.append(F.normalize(net(x).float() + net(x.flip(3)).float(), dim=1).cpu())
    return torch.cat(out)

def recall_at(Eq, oq, Eg, og, k):
    top = (Eq @ Eg.T).topk(k, dim=1).indices.numpy()
    return float(np.mean([oq[i] in og[top[i]] for i in range(len(oq))]))

def val_r1(net):                                                               # 1 ảnh/đối tượng làm gallery, còn lại query
    E, o = embed(net, X[iva]), obj[iva]; gi = np.array([np.where(o == c)[0][0] for c in va_obj]); qi = np.setdiff1d(np.arange(len(o)), gi)
    return recall_at(E[qi], o[qi], E[gi], o[gi], 1)

net = Net(80).to(DEV)
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X[itr]).permute(0, 3, 1, 2), torch.from_numpy(obj[itr]).long()),
                                 64, shuffle=True, drop_last=True, num_workers=0)
EP = 20
opt = torch.optim.AdamW([{"params": net.body.parameters(), "lr": 3e-4}, {"params": [*net.pool.parameters(), *net.emb.parameters(), net.W], "lr": 3e-3}], weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, [3e-4, 3e-3], total_steps=EP * len(dl), pct_start=0.1)
best, t0 = -1, time.time()
for ep in range(EP):
    net.train()
    for xb, yb in dl:
        xb = to_x(torch.stack([AUG(torch.rot90(x, random.randint(0, 3), (1, 2))) for x in xb])).to(DEV)
        with torch.autocast(DEV, torch.bfloat16, enabled=AMP):
            loss = F.cross_entropy(net.arcface(net(xb).float(), yb.to(DEV)), yb.to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    r1 = val_r1(net)                                                           # validation ngoài vòng batch
    if r1 > best: best = r1; torch.save(net.state_dict(), "best.pt")
    print(f"ep {ep} loss {loss.item():.2f} val R@1 {r1:.3f} ({time.time() - t0:.0f}s)", flush=True)

net.load_state_dict(torch.load("best.pt", map_location=DEV))
top = (embed(net, Xq) @ embed(net, Xg).T).topk(10, dim=1).indices.numpy()
sub["matches"] = [" ".join(g_ids[j] for j in row) for row in top]
sub.to_csv("submission.csv", index=False)
print("xong, best val R@1", round(best, 3))
