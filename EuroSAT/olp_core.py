# Tiện ích dùng chung: đọc ảnh có cache, chuẩn hoá, loader trên GPU, augmentation hình học trên GPU.
import os, json, random
import numpy as np, pandas as pd, torch, torch.nn.functional as F, cv2
from concurrent.futures import ThreadPoolExecutor
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def read_img(p, size):
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if x.ndim == 3 and x.shape[2] == 3: x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return x[..., None] if x.ndim == 2 else x


def load_imgs(paths, size, cache=None):
    """Ảnh uint8 dạng (N, C, H, W); lưu cache .npy cho lần chạy sau."""
    if cache and os.path.exists(cache): return np.load(cache)
    with ThreadPoolExecutor(os.cpu_count()) as ex: X = np.stack(list(ex.map(lambda p: read_img(p, size), paths)))
    X = np.ascontiguousarray(X.transpose(0, 3, 1, 2))
    if cache: np.save(cache, X)
    return X


class Norm:
    """Chuẩn hoá theo kênh, mean/std tính trên tập train."""
    def __init__(self, m=None, s=None): self.m, self.s = m, s

    def fit(self, X):
        x = torch.as_tensor(X[np.random.choice(len(X), min(len(X), 2000), replace=False)]).float()
        x = x.transpose(0, 1).flatten(1); self.m, self.s = x.mean(1).tolist(), x.std(1).tolist(); return self

    def __call__(self, x):
        v = (1, -1) + (1,) * (x.dim() - 2)
        return (x.float() - torch.tensor(self.m, device=x.device).view(v)) / torch.tensor(self.s, device=x.device).view(v)


def weights(y, src=None, ratio=None, alpha=0.5):
    """Trọng số lấy mẫu: cân bằng lớp (mũ alpha) và tỉ lệ giữa các nguồn dữ liệu src."""
    w = np.bincount(y)[y] ** -alpha * 1.0
    if src is not None: w *= np.array([ratio[s] / (src == s).sum() for s in range(src.max() + 1)])[src]
    return w / w.sum()


class Loader:
    """Toàn bộ dữ liệu nằm sẵn trên GPU, mỗi batch là một lát index (có thể lấy mẫu theo trọng số)."""
    def __init__(self, d, bs, train=True, w=None, tf=None):
        self.d = {k: torch.as_tensor(v).to(DEV) for k, v in d.items()}
        self.n = len(next(iter(self.d.values()))); self.bs, self.train, self.tf = bs, train, tf
        self.w = None if w is None else torch.as_tensor(w, dtype=torch.float, device=DEV)

    def __len__(self): return self.n // self.bs if self.train else -(-self.n // self.bs)

    def __iter__(self):
        if not self.train: idx = torch.arange(self.n, device=DEV)
        elif self.w is not None: idx = torch.multinomial(self.w, self.n, replacement=True)
        else: idx = torch.randperm(self.n, device=DEV)
        for i in range(len(self)):
            b = {k: v[idx[i * self.bs:(i + 1) * self.bs]] for k, v in self.d.items()}
            yield self.tf(b, self.train) if self.tf else b


def flip(*xs):
    """Lật ngang, lật dọc, chuyển vị ngẫu nhiên theo từng mẫu (đủ 8 phép của nhóm D4 với ảnh vuông)."""
    out = list(xs)
    for f in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.transpose(-1, -2)):
        r = torch.rand(len(xs[0]), device=xs[0].device) < 0.5
        out = [torch.where(r.view(-1, *[1] * (t.dim() - 1)), f(t), t) for t in out]
    return out


def theta(n, smin=0.6):
    """Ma trận affine cho random-resized-crop: tỉ lệ diện tích trong [smin, 1], vị trí ngẫu nhiên."""
    s = torch.empty(n, device=DEV).uniform_(smin, 1).sqrt()
    th = torch.zeros(n, 2, 3, device=DEV); th[:, 0, 0] = th[:, 1, 1] = s
    th[:, :, 2] = (torch.rand(n, 2, device=DEV) * 2 - 1) * (1 - s[:, None]); return th


def warp(x, th, mode="bilinear"):
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode=mode, padding_mode="reflection", align_corners=False)


def save_state(path, **kw): json.dump(kw, open(path, "w"), ensure_ascii=False)


def shift(Xtr, Xte):
    """Độ lệch trung bình từng kênh của test so với train, đơn vị độ lệch chuẩn của train."""
    a, b = Xtr.astype(np.float32), Xte.astype(np.float32)
    return np.round((b.mean((0, 2, 3)) - a.mean((0, 2, 3))) / a.std((0, 2, 3)), 3)
