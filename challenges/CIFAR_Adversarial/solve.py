"""Lời giải CIFAR_Adversarial: tấn công ensemble (mô hình A công bố + 3 mô hình thay thế tự train) bằng PGD có quán tính
(MI-FGSM) và đa dạng đầu vào (DI), eps = 8/255, rồi lượng tử hoá về uint8 hợp lệ.
Chạy: python solve.py <thư_mục_dữ_liệu> (chứa model_A.pt, mo_hinh.py, cifar_train.npz, test/, test_labels.csv, và private_test/ nếu đã giải nén)."""
import torch                                                   # Windows: nạp torch trước pandas
import base64, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from torch import nn
import torch.nn.functional as F
from PIL import Image

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
sys.path.insert(0, str(D))
from mo_hinh import tai_mo_hinh, MEAN, STD  # noqa: E402

A = tai_mo_hinh(D / "model_A.pt").to(DEV).eval()
d = np.load(D / "cifar_train.npz"); Xtr = torch.tensor(d["x"]).permute(0, 3, 1, 2).contiguous().to(DEV); Ytr = torch.tensor(d["y"]).long().to(DEV)
mean, std = torch.tensor(MEAN, device=DEV).view(1, 3, 1, 1), torch.tensor(STD, device=DEV).view(1, 3, 1, 1)


class ThayThe(nn.Module):
    """Mô hình thay thế: nhận ảnh 0-1 như A. rong = số kênh gốc (khác A để đa dạng)."""
    def __init__(self, rong=32, sau=3):
        super().__init__(); c, layers = 3, []
        for i in range(sau):
            co = rong * 2 ** i
            layers += [nn.Conv2d(c, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True), nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True), nn.MaxPool2d(2)]
            c = co
        self.f = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c, 10))
    def forward(self, x):
        return self.f((x - mean) / std)


def train_thay_the(seed, rong, EP=15, BS=256):
    seed_all(seed); m = ThayThe(rong).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=EP * (len(Xtr) // BS))
    for ep in range(EP):
        m.train()
        for idx in torch.randperm(len(Xtr), device=DEV).split(BS)[:len(Xtr) // BS]:
            xb = Xtr[idx].float() / 255
            if random.random() < 0.5: xb = xb.flip(3)
            xb = torch.roll(xb, tuple(int(v) for v in np.random.randint(-4, 5, 2)), (2, 3))
            loss = F.cross_entropy(m(xb), Ytr[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()


def pgd(models, x, y, eps=8 / 255, alpha=1.6 / 255, steps=20, mu=1.0):
    adv, g = x.clone(), torch.zeros_like(x)
    for _ in range(steps):
        adv.requires_grad_(True)
        inp = adv
        if random.random() < 0.7:                              # DI: thu nhỏ ngẫu nhiên rồi đệm về 32
            s = random.randint(27, 32); inp = F.interpolate(adv, size=(s, s), mode="bilinear", align_corners=False)
            p = 32 - s; l, t = random.randint(0, p), random.randint(0, p); inp = F.pad(inp, (l, p - l, t, p - t))
        loss = sum(F.cross_entropy(m(inp), y) for m in models)
        grad, = torch.autograd.grad(loss, adv)
        g = mu * g + grad / grad.abs().mean((1, 2, 3), keepdim=True).clamp(min=1e-12)   # MI-FGSM
        adv = torch.min(torch.max(adv.detach() + alpha * g.sign(), x - eps), x + eps).clamp(0, 1)
    return adv.detach()


def luong_tu(adv, x_u8):
    a = (adv * 255).round().short(); x = x_u8.short()
    return torch.min(torch.max(a, x - 8), x + 8).clamp(0, 255).to(torch.uint8)


t0 = time.time()
models = [A] + [train_thay_the(s, r) for s, r in [(1, 32), (2, 32), (3, 48)]]
print(f"đã train 3 mô hình thay thế ({(time.time() - t0) / 60:.1f} phút)")
nhan = pd.read_csv(D / "test_labels.csv"); thu = [(D / "test/images", nhan)]
for p in [Path("private_test"), D / "private_test"]:
    if (p / "test_labels.csv").exists():
        thu.append((p / "images", pd.read_csv(p / "test_labels.csv"))); break
ids, out = [], []
for thu_muc, nh in thu:
    x_u8 = torch.tensor(np.stack([np.asarray(Image.open(thu_muc / f"{i}.png").convert("RGB")) for i in nh.id])).permute(0, 3, 1, 2).to(DEV)
    y = torch.tensor(nh.label.values, device=DEV)
    for b in range(0, len(y), 250):
        adv = luong_tu(pgd(models, x_u8[b:b + 250].float() / 255, y[b:b + 250]), x_u8[b:b + 250])
        with torch.no_grad():
            print(f"  lô {b}: lừa A {(A(adv.float() / 255).argmax(1) != y[b:b + 250]).float().mean():.3f}, "
                  f"lừa mô hình thay thế cuối {(models[-1](adv.float() / 255).argmax(1) != y[b:b + 250]).float().mean():.3f}")
        out += [base64.b64encode(a.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).decode() for a in adv]
    ids += nh.id.tolist()
pd.DataFrame({"id": ids, "image": out}).to_csv("submission.csv", index=False)
print(f"đã ghi submission.csv: {len(ids)} ảnh ({(time.time() - t0) / 60:.1f} phút)")
