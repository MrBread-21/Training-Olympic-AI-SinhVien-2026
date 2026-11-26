import torch                                                   # Windows: nạp torch trước pandas
import random
import numpy as np, pandas as pd
from torch import nn
import torch.nn.functional as F
from scipy.io import wavfile
from sklearn.metrics import f1_score

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000

def doc(p):
    sr, a = wavfile.read(p); a = a.astype(np.float32) / 32768
    return np.pad(a, (0, max(0, SR - len(a))))[:SR]

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
LOP = sorted(tr.label.unique()); y_all = torch.tensor([LOP.index(l) for l in tr.label], device=DEV)
X_all = torch.tensor(np.stack([doc(f"data/train/{i}.wav") for i in tr.id]), device=DEV)       # (N, 16000) nạp hết lên GPU
X_te = torch.tensor(np.stack([doc(f"data/test/{i}.wav") for i in te.id]), device=DEV)

def mel_fb(n_mel=64, n_fft=512):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    b = np.floor((n_fft + 1) * hz(np.linspace(0, 2595 * np.log10(1 + SR / 2 / 700), n_mel + 2)) / SR).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1), np.float32)
    for i in range(n_mel):
        fb[i, b[i]:b[i + 1]] = (np.arange(b[i], b[i + 1]) - b[i]) / max(1, b[i + 1] - b[i])
        fb[i, b[i + 1]:b[i + 2]] = (b[i + 2] - np.arange(b[i + 1], b[i + 2])) / max(1, b[i + 2] - b[i + 1])
    return torch.tensor(fb, device=DEV)
FB, WIN = mel_fb(), torch.hann_window(400, device=DEV)

def logmel(x):                                                 # (B, 16000) -> (B, 1, 64, 101)
    S = torch.stft(x, 512, hop_length=160, win_length=400, window=WIN, return_complex=True).abs() ** 2
    return torch.log(FB @ S + 1e-6).unsqueeze(1)

def nhieu_mau(B, kho):                                          # trắng / hồng / tiếng nói khác trong train
    n = torch.randn(B, SR, device=DEV)
    f = torch.fft.rfftfreq(SR, device=DEV); f[0] = f[1]
    hong = torch.fft.irfft(torch.fft.rfft(n) / f.sqrt(), SR)
    noi = kho[torch.randint(0, len(kho), (B, 3), device=DEV)].sum(1)
    k = torch.randint(0, 3, (B, 1), device=DEV)
    return torch.where(k == 0, n, torch.where(k == 1, hong, noi))

def aug(x, kho):
    B = len(x)
    x = torch.stack([torch.roll(a, int(s)) for a, s in zip(x, torch.randint(-1600, 1601, (B,)))])   # lệch thời gian ±0,1 s
    n = nhieu_mau(B, kho); snr = torch.empty(B, 1, device=DEV).uniform_(-2, 20)
    px, pn = x.pow(2).mean(1, keepdim=True) + 1e-8, n.pow(2).mean(1, keepdim=True) + 1e-8
    co = (torch.rand(B, 1, device=DEV) < 0.8).float()                                               # 80% có nhiễu, giống test
    x = x + co * n * torch.sqrt(px / pn / 10 ** (snr / 10))
    x = x / x.abs().amax(1, keepdim=True).clamp(min=1e-4) * torch.empty(B, 1, device=DEV).uniform_(0.2, 0.9)
    S = logmel(x)
    for _ in range(2):                                                                               # SpecAugment: che dải tần, che đoạn thời gian
        f0 = random.randint(0, 54); t0 = random.randint(0, 81)
        S[:, :, f0:f0 + random.randint(0, 10)] = S.mean(); S[:, :, :, t0:t0 + random.randint(0, 20)] = S.mean()
    return S

def khoi(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                         nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True), nn.MaxPool2d(2))
def make_model():
    return nn.Sequential(nn.BatchNorm2d(1), khoi(1, 32), khoi(32, 64), khoi(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, len(LOP))).to(DEV)

def train(X, y, EP=30, BS=64):
    model = make_model(); kho = X
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EP * ((len(X) + BS - 1) // BS))
    for ep in range(EP):
        model.train()
        for idx in torch.randperm(len(X), device=DEV).split(BS):
            loss = F.cross_entropy(model(aug(X[idx], kho)), y[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return model

@torch.inference_mode()
def predict(model, X):
    model.eval()
    return torch.cat([model(logmel(x)).softmax(1) for x in X.split(256)]).cpu().numpy()

# val theo người nói, làm nhiễu giống test (seed cố định)
spk = tr.speaker.unique(); vs = set(np.random.default_rng(0).choice(spk, 8, replace=False))
mv = torch.tensor(tr.speaker.isin(vs).values, device=DEV)
m = train(X_all[~mv], y_all[~mv])
torch.manual_seed(1); Xv = X_all[mv].clone()
with torch.no_grad():
    n = nhieu_mau(len(Xv), X_all[~mv]); snr = torch.empty(len(Xv), 1, device=DEV).uniform_(0, 15)
    Xv = Xv + n * torch.sqrt((Xv.pow(2).mean(1, keepdim=True) + 1e-8) / (n.pow(2).mean(1, keepdim=True) + 1e-8) / 10 ** (snr / 10))
pv = predict(m, Xv).argmax(1)
print(f"val macro-F1 (người nói mới, có nhiễu) = {f1_score(y_all[mv].cpu(), pv, average='macro'):.4f}")

seed_all(42); m = train(X_all, y_all)                          # bản cuối: toàn bộ train
pd.DataFrame({"id": te.id, "label": [LOP[i] for i in predict(m, X_te).argmax(1)]}).to_csv("submission.csv", index=False)
print("đã ghi submission.csv")
