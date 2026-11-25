"""Nordland VPR - train nhanh có giới hạn thời gian (mặc định 25 phút train, tổng ~28 phút).

ResNet18 ImageNet (torchvision DEFAULT) tới layer3 + descriptor lưới không gian 4x7, loss InfoNCE
hai chiều đông<->hè, batch gồm các cụm khung liên tiếp (hard negative). LR cosine theo thời gian.

Tốc độ: đĩa máy này đọc file lẻ ~10 ms/ảnh, nên nạp toàn bộ byte JPEG một lần (từ zip nếu có,
~10 giây, ~750MB RAM) rồi giải nén + resize + augment trên GPU (nvjpeg). Không dùng DataLoader.

Chạy:
    python train_fast.py                     # train 85% đầu, val 15% cuối, in R@1/5/10, ghi submission.csv
    python train_fast.py --full              # train toàn bộ train (không val), cho bài nộp cuối
    python train_fast.py --minutes 10 --arch resnet34
"""
import argparse
import math
import random
import time
import zipfile
from pathlib import Path

import torch  # phải import torch TRƯỚC numpy/pandas, nếu không lỗi DLL c10.dll trên máy này
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.io import decode_jpeg, read_file
import numpy as np
import pandas as pd

H, W = 128, 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------- dữ liệu ----------------
def load_bytes(root, zip_path, names):
    """names: list đường dẫn tương đối 'train/summer/00000.jpg'. Trả về list tensor uint8 (byte JPEG)."""
    t = time.time()
    if zip_path is not None and Path(zip_path).exists():
        with zipfile.ZipFile(zip_path) as z:
            out = [torch.frombuffer(bytearray(z.read(n)), dtype=torch.uint8) for n in names]
        src = "zip"
    else:
        out = [read_file(str(root / n)) for n in names]
        src = "thư mục"
    print(f"nạp {len(out)} ảnh từ {src} trong {time.time()-t:.1f}s", flush=True)
    return out


def decode(byte_list, dev):
    """Giải nén batch JPEG trên GPU, resize về HxW, trả float [0,1] Bx3xHxW."""
    xs = decode_jpeg(byte_list, device=dev)
    x = torch.stack(xs).float().div_(255)
    return F.interpolate(x, size=(H, W), mode="bilinear", antialias=True, align_corners=False)


def normalize(x):
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous()


def hard_batch(n, batch, rng, chunk=8):
    """batch/chunk cụm; mỗi cụm = chunk khung liên tiếp cách nhau 3-6 khung (hard negative thật:
    gần giống nhau nhưng lệch >2 khung nên tính là sai)."""
    idx = set()
    while len(idx) < batch:
        a, k = rng.randrange(n), 0
        while k < chunk and a < n and len(idx) < batch:
            idx.add(a); k += 1
            a += rng.randint(3, 6)
    return sorted(idx)


def gpu_augment(w, s, gen):
    """Màu độc lập từng ảnh, hình học nhẹ độc lập, lật ngang chung cho cả cặp."""
    B, dev = w.shape[0], w.device

    def r(*shape):
        return torch.rand(*shape, device=dev, generator=gen)

    def color(x):
        b, c, sat = 1 + (r(B, 1, 1, 1) - 0.5) * 0.6, 1 + (r(B, 1, 1, 1) - 0.5) * 0.6, 1 + (r(B, 1, 1, 1) - 0.5) * 0.8
        gray = x.mean(1, keepdim=True)
        x = gray + (x - gray) * sat
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * c + m * b
        g = r(B, 1, 1, 1) < 0.2  # đôi khi bỏ màu
        return torch.where(g, x.mean(1, keepdim=True).expand_as(x), x).clamp(0, 1)

    def geom(x):
        sc, tx, ty = 0.88 + r(B) * 0.12, (r(B) - 0.5) * 0.12, (r(B) - 0.5) * 0.08
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0], th[:, 1, 1], th[:, 0, 2], th[:, 1, 2] = sc, sc, tx, ty
        grid = F.affine_grid(th, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    w, s = geom(color(w)), geom(color(s))
    flip = r(B, 1, 1, 1) < 0.5
    return torch.where(flip, w.flip(-1), w), torch.where(flip, s.flip(-1), s)


# ---------------- model + loss ----------------
class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x):
        return x.clamp(min=1e-6).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class Net(nn.Module):
    """head='grid': lấy layer3 (stride 16), conv1x1 giảm kênh, pool về lưới gh x gw rồi làm phẳng:
    giữ bố cục không gian (camera cố định ở đầu tàu nên bố cục rất ổn định giữa các mùa).
    head='gem': layer4 + GeM + Linear (descriptor toàn cục kiểu cũ)."""

    def __init__(self, arch, head="grid", ch=128, grid=(4, 7), dim=512):
        super().__init__()
        m = getattr(tv.models, arch)(weights="DEFAULT")
        layers = [m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
        if head == "gem":
            layers.append(m.layer4)
        self.body = nn.Sequential(*layers)
        for mod in [m.conv1, m.bn1, m.layer1]:  # đóng băng tầng đầu: nhanh hơn, ít overfit
            for p in mod.parameters():
                p.requires_grad = False
        with torch.no_grad():
            c = self.body(torch.zeros(1, 3, H, W)).shape[1]  # số kênh ra của tầng cuối
        self.head = head
        if head == "grid":
            self.red = nn.Conv2d(c, ch, 1)
            self.grid = grid
        else:
            self.pool, self.fc = GeM(), nn.Linear(c, dim)

    def train(self, mode=True):
        """BatchNorm luôn ở eval (dùng running stats ImageNet). Batch toàn khung liên tiếp gần giống nhau
        nên thống kê batch lệch hẳn running stats; không đóng băng thì val sụp về ~0 sau vài phút."""
        super().train(mode)
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        return self

    def forward(self, x):
        f = self.body(x)
        if self.head == "grid":
            z = F.adaptive_avg_pool2d(self.red(f), self.grid).flatten(1)
        else:
            z = self.fc(self.pool(f))
        return F.normalize(z, dim=1)


def info_nce(zw, zs, fr, tau):
    sim = zw @ zs.T / tau
    d = (fr[:, None] - fr[None, :]).abs()
    eye = torch.eye(len(fr), dtype=torch.bool, device=fr.device)
    sim = sim.masked_fill((d <= 2) & ~eye, float("-inf"))  # khung lệch <=2 không tính là negative
    y = torch.arange(len(fr), device=fr.device)
    return 0.5 * (F.cross_entropy(sim, y) + F.cross_entropy(sim.T, y))


@torch.no_grad()
def embed(model, byte_list, dev, bs=128, tta=True):
    model.eval()
    out = []
    for i in range(0, len(byte_list), bs):
        x = normalize(decode(byte_list[i:i + bs], dev))
        z = model(x)
        if tta:
            z = F.normalize(z + model(x.flip(-1)), dim=1)
        out.append(z)
    model.train()
    return torch.cat(out)


def recall(sim, q_fr, d_fr, ks=(1, 5, 10)):
    top = sim.topk(max(ks), dim=1).indices
    ok = (d_fr[top] - q_fr[:, None]).abs() <= 2
    return {f"R@{k}": round(ok[:, :k].any(1).float().mean().item(), 4) for k in ks}


# ---------------- main ----------------
def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here.parents[1] / "selfhost_kaggle/Nordland_PlaceRecognition/upload"))
    ap.add_argument("--zip", default=str(here.parents[1] / "selfhost_kaggle/Nordland_PlaceRecognition/Nordland_PlaceRecognition_data.zip"), help="zip dữ liệu (nhanh hơn đọc file lẻ); bỏ qua nếu không có")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--head", default="grid", choices=["grid", "gem"])
    ap.add_argument("--minutes", type=float, default=25.0, help="ngân sách thời gian cho vòng train")
    ap.add_argument("--batch", type=int, default=64, help="số cặp đông-hè mỗi batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--full", action="store_true", help="train toàn bộ train, bỏ val")
    ap.add_argument("--eval_every", type=float, default=5.0, help="phút giữa các lần đo val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    rng = random.Random(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    root = Path(a.data)
    df = pd.read_csv(root / "train.csv").sort_values("frame").reset_index(drop=True)
    q = pd.read_csv(root / "query.csv"); db = pd.read_csv(root / "database.csv")

    # nạp byte một lần theo đúng thứ tự trong zip (đọc tuần tự)
    names = list(df.summer_path) + list(df.winter_path) + [f"test/query/{i}.jpg" for i in q.id] + [f"test/database/{i}.jpg" for i in db.id]
    allb = load_bytes(root, a.zip, names)
    n = len(df)
    S, Wn = allb[:n], allb[n:2 * n]
    Q, D = allb[2 * n:2 * n + len(q)], allb[2 * n + len(q):]

    if a.full:
        tr = np.arange(n); va = None
    else:  # val = 15% cuối hành trình, bỏ 50 khung đệm để không rò rỉ
        cut = int(n * 0.85)
        tr, va = np.arange(cut - 50), np.arange(cut, n)
    print(f"train {len(tr)} cặp" + ("" if va is None else f", val {len(va)} cặp"), flush=True)
    frames = torch.tensor(df.frame.values, device=dev)

    model = Net(a.arch, a.head).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)

    def run_val():
        if va is None:
            return None
        zq = embed(model, [Wn[i] for i in va], dev)
        zd = embed(model, [S[i] for i in va], dev)
        fr = frames[va]
        return recall(zq @ zd.T, fr, fr)

    budget = a.minutes * 60
    t0 = time.time(); next_eval = a.eval_every * 60
    step, loss_ema = 0, None
    while True:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget  # warmup 3% rồi cosine theo thời gian
        lr = a.lr * (frac / 0.03 if frac < 0.03 else 0.5 * (1 + math.cos(math.pi * (frac - 0.03) / 0.97)))
        for g in opt.param_groups:
            g["lr"] = lr

        idx = tr[hard_batch(len(tr), a.batch, rng)]
        w, s = gpu_augment(decode([Wn[i] for i in idx], dev), decode([S[i] for i in idx], dev), gen)
        z = model(normalize(torch.cat([w, s])))
        fr = frames[idx]
        loss = info_nce(z[:len(idx)], z[len(idx):], fr, a.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if step % 50 == 0:
            ips = step * 2 * a.batch / (time.time() - t0)
            print(f"[{el/60:5.1f}m] step {step} loss {loss_ema:.3f} lr {lr:.2e} {ips:.0f} img/s", flush=True)
        if va is not None and el > next_eval:
            next_eval += a.eval_every * 60
            print(f"   VAL {run_val()}", flush=True)

    print(f"train xong: {step} bước, {(time.time()-t0)/60:.1f} phút", flush=True)
    if va is not None:
        print(f"VAL cuối {run_val()}", flush=True)
    torch.save(model.state_dict(), Path(a.out).with_suffix(".pt"))

    top = (embed(model, Q, dev) @ embed(model, D, dev).T).topk(10, dim=1).indices.cpu().numpy()
    ids = db.id.values
    pd.DataFrame({"id": q.id, "predictions": [" ".join(ids[t]) for t in top]}).to_csv(a.out, index=False)
    print(f"ghi {a.out} ({len(q)} dòng), tổng {(time.time()-t0)/60:.1f} phút", flush=True)


if __name__ == "__main__":
    main()
