# EuroSAT bản 2: ResNet-18 train từ đầu, augmentation mô phỏng biến đổi của test, chọn model trên val bị làm hỏng,
# pseudo-label ảnh unlabeled, TTA 8 phép xoay/lật. Chạy: python train_img.py (cần olp_core.py cùng thư mục).
import copy, time, torchvision, torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from olp_core import *
C = dict(size=64, bs=128, ep1=15, ep2=10, lr=2e-3, wd=0.05, ema=0.999, conf=0.8, min_pl=300, n_jpeg=2, maxpool=True)
seed(42); os.makedirs("cache", exist_ok=True)
classes = pd.read_csv("classes.csv").label.tolist(); ncls = len(classes)
AMP = DEV.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8      # bf16 chỉ trên GPU Ampere trở lên, còn lại fp32
print(torch.cuda.get_device_name() if DEV.type == "cuda" else "CPU", "| amp bf16" if AMP else "| fp32")

def files(d): return {p.stem: str(p) for p in Path(d).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
def split(d, csv):
    df = pd.read_csv(csv); fm = files(d)
    return load_imgs([fm[i] for i in df.id], C["size"], f"cache/{d}{C['size']}.npy"), df.label.map(classes.index).values

def jpeg(X, seed, q=(10, 60)):                                        # nén JPEG offline (CPU), chất lượng ngẫu nhiên
    qs = np.random.default_rng(seed).integers(*q, len(X))
    def f(a, qq):
        _, e = cv2.imencode(".jpg", np.ascontiguousarray(a.transpose(1, 2, 0)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(qq)])
        return np.ascontiguousarray(cv2.imdecode(e, 1)[..., ::-1].transpose(2, 0, 1))
    with ThreadPoolExecutor() as ex: return np.stack(list(ex.map(f, X, qs)))

def blur(x, s):                                                       # Gaussian blur 5x5, depthwise
    r = torch.arange(-2, 3, device=x.device).float(); k = torch.exp(-r ** 2 / (2 * s * s)); k = k / k.sum()
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), (k[:, None] * k[None]).expand(x.shape[1], 1, 5, 5).contiguous(), groups=x.shape[1])

def corrupt(x, p=0.5):                                                # x trong [0,1]: lệch màu, bão hoà, mờ, nhiễu
    B, d = len(x), x.device; on = lambda: (torch.rand(B, 1, 1, 1, device=d) < p).float()
    g = 1 + (torch.rand(B, 3, 1, 1, device=d) * 2 - 1) * 0.3; b = (torch.rand(B, 3, 1, 1, device=d) * 2 - 1) * 0.1
    m = on(); x = m * (x * g + b) + (1 - m) * x
    gr = x.mean(1, keepdim=True); s = 0.3 + torch.rand(B, 1, 1, 1, device=d); m = on(); x = m * (gr + (x - gr) * s) + (1 - m) * x
    r = torch.rand(B, 1, 1, 1, device=d); x = torch.where(r < p / 2, blur(x, 0.8), torch.where(r < p, blur(x, 1.5), x))
    m = on(); x = x + m * torch.rand(B, 1, 1, 1, device=d) * 0.1 * torch.randn_like(x)
    return x.clamp(0, 1)

def make_tf(norm):
    def tf(b, train):
        x = b["x"].float() / 255
        if train: x, = flip(x); x = corrupt(warp(x, theta(len(x), 0.7)))   # flip() gồm cả chuyển vị nên đủ 8 phép D4
        out = {"x": norm(x * 255)}
        if "y" in b: out["y"] = b["y"]; out["ys"] = F.one_hot(b["y"].long(), ncls).float() * 0.9 + 0.01
        return out
    return tf

@torch.no_grad()
def tta8(model, loader, tta=True):                                    # trung bình xác suất qua 8 phép D4
    model.eval(); P = []
    for b in loader:
        x = b["x"]; vs = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)] if tta else [x]
        if tta: vs += [v.transpose(-1, -2) for v in vs]
        with torch.autocast(DEV.type, torch.bfloat16, enabled=AMP): P.append(sum(model(v).float().softmax(1) for v in vs).cpu() / len(vs))
    P = torch.cat(P); assert not P.isnan().any(), "dự đoán có NaN"
    return P

def mf1(model, loader, y, tta=False):
    p = tta8(model, loader, tta).argmax(1).numpy()
    return f1_score(y, p, average="macro")

def train(X, y, w, norm, vl, yv, ep, name):
    m = torchvision.models.resnet18(weights=None, num_classes=ncls, zero_init_residual=True)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    if not C["maxpool"]: m.maxpool = nn.Identity()                       # bỏ maxpool: tốt hơn một chút nhưng chậm khoảng 4 lần
    model = m.to(DEV).to(memory_format=torch.channels_last); ema = copy.deepcopy(model).eval()
    tl = Loader({"x": X, "y": y}, C["bs"], w=w, tf=make_tf(norm))
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": C["wd"]},
                             {"params": [p for p in model.parameters() if p.ndim <= 1], "weight_decay": 0}], C["lr"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, C["lr"], total_steps=ep * len(tl), pct_start=0.2)
    best, t0 = -1, time.time()
    for e in range(ep):
        model.train()
        for b in tl:
            with torch.autocast(DEV.type, torch.bfloat16, enabled=AMP):
                loss = F.cross_entropy(model(b["x"].contiguous(memory_format=torch.channels_last)).float(), b["ys"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            d = min(C["ema"], (1 + sch.last_epoch) / (10 + sch.last_epoch))
            with torch.no_grad():
                for a, p in zip(ema.state_dict().values(), model.state_dict().values()):
                    a.copy_(a * d + p * (1 - d)) if a.dtype.is_floating_point else a.copy_(p)
        if e % 5 == 4 or e == ep - 1:                                  # chọn checkpoint theo macro-F1 trên val bị làm hỏng
            f = mf1(ema, vl, yv)
            if f > best: best = f; torch.save(ema.state_dict(), f"{name}.pt")
            print(f"[{name}] ep {e} loss {loss.item():.3f} valC-F1 {f:.4f} | {time.time() - t0:.0f}s", flush=True)
    ema.load_state_dict(torch.load(f"{name}.pt")); return ema

# ---- dữ liệu: train/val sạch và một bản val bị làm hỏng cố định (seed 123) ----
Xtr, ytr = split("train", "train_labels.csv"); Xva, yva = split("val", "val_labels.csv")
norm = Norm().fit(Xtr); save_state("eurosat_v2.json", m=norm.m, s=norm.s, C=C)
torch.manual_seed(123)
xc = corrupt(torch.as_tensor(jpeg(Xva, 99)).to(DEV).float() / 255, p=0.6); xc, = flip(xc)
Xvc = (xc * 255).round().byte().cpu().numpy()
vl = Loader({"x": Xvc}, 512, train=False, tf=make_tf(norm)); vl_clean = Loader({"x": Xva}, 512, train=False, tf=make_tf(norm))

def with_jpeg(X, y, s0):                                             # nguồn 0 = sạch, nguồn 1 = các bản nén JPEG
    Xs = [X] + [jpeg(X, s0 + i) for i in range(C["n_jpeg"])]; src = np.repeat(np.arange(len(Xs)) > 0, len(X)).astype(int)
    return np.concatenate(Xs), np.tile(y, len(Xs)), weights(np.tile(y, len(Xs)), src, {0: 0.5, 1: 0.5}, alpha=0.5)

# ---- giai đoạn 1: chỉ dữ liệu có nhãn ----
X1, y1, w1 = with_jpeg(Xtr, ytr, 0)
m1 = train(X1, y1, w1, norm, vl, yva, C["ep1"], "stage1")
print("stage1 | val sạch F1", round(mf1(m1, vl_clean, yva), 4), "| val bẩn F1 + TTA8", round(mf1(m1, vl, yva, True), 4))

# ---- giai đoạn 2: pseudo-label ảnh unlabeled có độ tin cậy > conf ----
fu = files("unlabeled"); Xu = load_imgs(list(fu.values()), C["size"], f"cache/unl{C['size']}.npy")
pu = tta8(m1, Loader({"x": Xu}, 512, train=False, tf=make_tf(norm))); conf, yu = pu.max(1)
keep = (conf > C["conf"]).numpy(); print("pseudo-label giữ", keep.sum(), "/", len(Xu), "| theo lớp", np.bincount(yu.numpy()[keep], minlength=ncls))
f1v, f2v, m2 = mf1(m1, vl, yva, True), -1, None
if keep.sum() >= C["min_pl"]:                                          # quá ít ảnh tin cậy thì bỏ qua giai đoạn 2
    X2, y2, w2 = with_jpeg(np.concatenate([Xtr, Xu[keep]]), np.concatenate([ytr, yu.numpy()[keep]]), 10)
    m2 = train(X2, y2, w2, norm, vl, yva, C["ep2"], "stage2"); f2v = mf1(m2, vl, yva, True)
else: print("bỏ qua stage2: cần >=", C["min_pl"], "ảnh pseudo-label")
print("val bẩn F1 + TTA8 | stage1", round(f1v, 4), "| stage2", round(f2v, 4))

# ---- test: TTA8, ghép model giai đoạn 1 nếu không kém giai đoạn 2 quá 0.02 ----
sub = pd.read_csv("sample_submission.csv"); ft = files("test")
Xte = load_imgs([ft[i] for i in sub.id], C["size"], f"cache/test{C['size']}.npy")
print("test", Xte.shape, Xte.dtype, "min/max", Xte.min(), Xte.max(), "| lệch so với train:", shift(Xtr, Xte))
assert Xte.dtype == Xtr.dtype and Xte.shape[1:] == Xtr.shape[1:], "ảnh test khác định dạng ảnh train"
tl_ = Loader({"x": Xte}, 512, train=False, tf=make_tf(norm))
P = tta8(m1, tl_) if m2 is None else tta8(m2, tl_) + (tta8(m1, tl_) if f1v > f2v - 0.02 else 0)
sub["label"] = [classes[i] for i in P.argmax(1).tolist()]; vc = sub.label.value_counts()
print("phân bố dự đoán test:", vc.to_dict())
assert vc.iloc[0] < 0.5 * len(sub), "hơn 50% ảnh test dồn vào một lớp, cần kiểm tra lại"
sub.to_csv("submission.csv", index=False); print("đã ghi submission.csv")