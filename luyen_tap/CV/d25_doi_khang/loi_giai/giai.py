import os, random
import torch, torch.nn.functional as F
import cv2, numpy as np, pandas as pd
from mo_hinh import nap

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
m = nap("mo_hinh.pt", DEV)                                               # đã ở chế độ eval()
for p in m.parameters(): p.requires_grad_(False)
te = pd.read_csv("data/test.csv"); CLS = pd.read_csv("data/classes.csv").label.tolist()
U8 = np.stack([cv2.cvtColor(cv2.imread(f"data/test/{i}.png"), cv2.COLOR_BGR2RGB) for i in te.id])
X = torch.from_numpy(U8).permute(0, 3, 1, 2).float().div(255).to(DEV)
y = torch.tensor(te.label.map(CLS.index).values, device=DEV)
EPS, ALPHA, STEPS = 8 / 255, 2 / 255, 10

def grad_sign(x, y):
    x = x.clone().requires_grad_(True)
    F.cross_entropy(m(x), y).backward()
    return x.grad.sign()

def fgsm(x, y): return (x + EPS * grad_sign(x, y)).clamp(0, 1)

def pgd(x, y):
    xa = (x + torch.empty_like(x).uniform_(-EPS, EPS)).clamp(0, 1)          # khởi đầu ngẫu nhiên trong quả cầu L_inf
    for _ in range(STEPS):
        xa = xa + ALPHA * grad_sign(xa, y)
        xa = torch.min(torch.max(xa, x - EPS), x + EPS).clamp(0, 1)          # chiếu về quả cầu L_inf rồi về [0, 1]
    return xa

@torch.no_grad()
def predict(x):
    m.eval()
    with torch.inference_mode():
        return torch.cat([m(b).argmax(1) for b in x.split(64)])

def to_u8(xa):                                                         # lượng tử hoá, giữ |lệch| <= 8 sau khi làm tròn
    a = np.rint(xa.detach().permute(0, 2, 3, 1).cpu().numpy() * 255)
    return np.clip(a, U8.astype(int) - 8, U8.astype(int) + 8).clip(0, 255).astype(np.uint8)

print("FGSM thành công:", (predict(fgsm(X, y)) != y).float().mean().item())
adv = to_u8(pgd(X, y))
os.makedirs("adv", exist_ok=True)
for i, a in zip(te.id, adv): cv2.imwrite(f"adv/{i}.png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
back = np.stack([cv2.cvtColor(cv2.imread(f"adv/{i}.png"), cv2.COLOR_BGR2RGB) for i in te.id])   # đọc lại PNG
xb = torch.from_numpy(back).permute(0, 3, 1, 2).float().div(255).to(DEV)
mse = ((back.astype(float) - U8) ** 2).mean((1, 2, 3))
print("PGD thành công sau PNG:", (predict(xb) != y).float().mean().item(),
      "| PSNR TB %.2f dB | lệch max %d" % ((10 * np.log10(255 ** 2 / mse)).mean(), np.abs(back.astype(int) - U8).max()))

# Grad-CAM trên lớp conv cuối (features), cho lớp đúng
f = m.features((X - 0.5) / 0.25).detach().requires_grad_(True)
score = m.head(f.mean((2, 3))).gather(1, y[:, None]).sum()
g, = torch.autograd.grad(score, f)
cam = F.relu((g.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True))
cam = F.interpolate(cam, size=64, mode="bilinear", align_corners=False)[:, 0]
cam = cam / (cam.flatten(1).max(1).values[:, None, None] + 1e-8)
np.save("cam.npy", cam.detach().cpu().numpy().astype(np.float32))
print("đã ghi adv/ (100 PNG) và cam.npy", tuple(cam.shape))
