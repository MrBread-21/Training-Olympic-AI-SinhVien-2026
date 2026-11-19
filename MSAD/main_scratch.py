"""MSAD, bản không dùng trọng số pretrained: mọi mô hình train từ đầu trên dữ liệu của bài, seed cố định.

  python main_scratch.py               ghi submission.csv (khoảng 15 phút trên GTX 1650, gồm cả đọc video)
  python main_scratch.py --cv          điểm OOF 5-fold của từng thành phần và ensemble
  python main_scratch.py --cv --ablate so sánh có và không có đặc trưng cửa sổ sự kiện (bật khi nộp bằng --win)

Pipeline
 1. Đọc mỗi video một lần, 48 cặp khung (t, t + 0.2 giây) lấy đều:
      - 128x72: 55 đặc trưng thủ công (optical flow, hướng, lưới 3x3, divergence, màu HSV, lửa/nước, nền)
      - 112x112: RGB và (flow x, flow y, hiệu ảnh xám) cho CNN
 2. Mô hình bảng, 2 tập cột (A đầy đủ, B tập con): XGB + LogReg + ExtraTrees để phân loại,
    XGB theo khung có điều kiện lớp để định vị.
 3. CNN two-stream train từ đầu (ResNet + SE + TSM cho RGB, luồng chuyển động, TCN theo thời gian; đầu khung và đầu video).
 4. Ensemble: p = 0.8 * (pA + pB) / 2 + 0.2 * pCNN, nhân P(Normal) với 0.8; đoạn sự kiện = Kadane trên xác suất khung, ngưỡng 0.4.
OOF 5-fold: ensemble 0.283 (chỉ bảng A 0.245, B 0.223, CNN 0.205). Public 0.415 với bản 2 mô hình bảng.
"""
import argparse, copy, csv, math, os, random, sys, time
from collections import Counter
from multiprocessing import Pool
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import scorer  # noqa: E402
sys.stdout.reconfigure(encoding='utf-8')

L = list(scorer.MSAD_LABELS); K = len(L)               # Normal = lớp 0
T, FW, FH, SZ = 48, 128, 72, 112                        # số vị trí khung; kích thước cho đặc trưng / cho CNN
OUT, STRIDE = 96, 3; TS = T // STRIDE                   # CNN chạy 96px, mỗi lượt 16 khung (cách 3)
W_CNN, NSCALE, THR = 0.2, 0.8, 0.4                      # chọn bằng OOF 5-fold (xem --cv)
CACHE, CNN_OOF = 'outputs/main_cache.npz', 'outputs/cnn_oof.npz'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def rows(p):
    with open(p, encoding='utf-8') as f: return list(csv.DictReader(f))


INFO = {r['video_id']: r for r in rows('video_info.csv')}


# ============================================================================ 1. đọc video (1 lượt)
NAMES = (['mag_mean', 'mag_std', 'mag_p50', 'mag_p90', 'mag_p99', 'mag_gt1', 'mag_gt3', 'dx', 'dy', 'dy_top', 'adx_top']
         + [f'dir{i}' for i in range(8)] + [f'grid{i}' for i in range(9)] + ['div_mean', 'div_p90', 'diff_mean', 'diff_gt25']
         + [f'hue{i}' for i in range(12)] + ['v_mean', 'v_std', 's_mean', 'bright', 'dark', 'fire', 'blue', 'edges']
         + ['bg_diff', 'bg_fg', 'prev_diff'])
SHORT = [NAMES.index(n) for n in ['mag_p50', 'mag_p90', 'mag_p99', 'mag_mean', 'dx', 'dy', *[f'grid{i}' for i in range(9)],
                                  *[f'dir{i}' for i in range(8)], *[f'hue{i}' for i in range(12)],
                                  'v_mean', 'v_std', 's_mean', 'fire', 'blue', 'diff_mean']]


def frame_feats(a, b):
    g0, g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 13, 3, 5, 1.1, 0)
    dx, dy = fl[..., 0], fl[..., 1]; mag = np.sqrt(dx * dx + dy * dy); ang = np.arctan2(dy, dx); tot = mag.sum() + 1e-6
    m90 = mag > np.percentile(mag, 90)
    f = [mag.mean(), mag.std(), *np.percentile(mag, [50, 90, 99]), (mag > 1).mean(), (mag > 3).mean(), dx.mean(), dy.mean(),
         dy[m90].mean() if m90.any() else 0, np.abs(dx[m90]).mean() if m90.any() else 0,
         *(np.histogram(ang, 8, (-np.pi, np.pi), weights=mag)[0] / tot),
         *[mag[i * FH // 3:(i + 1) * FH // 3, j * FW // 3:(j + 1) * FW // 3].mean() for i in range(3) for j in range(3)]]
    div = np.gradient(dx, axis=1) + np.gradient(dy, axis=0); d = cv2.absdiff(g0, g1).astype(np.float32)
    h, s, v = [x.astype(np.float32) for x in cv2.split(cv2.cvtColor(a, cv2.COLOR_BGR2HSV))]
    f += [div.mean(), np.percentile(np.abs(div), 90), d.mean(), (d > 25).mean(),
          *(np.histogram(h[(s > 60) & (v > 60)], 12, (0, 180))[0] / h.size),
          v.mean(), v.std(), s.mean(), (v > 230).mean(), (v < 40).mean(),
          ((h < 25) & (s > 100) & (v > 150)).mean(), ((h > 90) & (h < 130) & (s > 50)).mean(), (cv2.Canny(g0, 80, 160) > 0).mean()]
    return f, g0


def read_video(vid):
    n, lag = int(INFO[vid]['frames']), max(1, round(float(INFO[vid]['fps']) / 6))
    pos = np.linspace(0, max(n - 1 - lag, 0), T).round().astype(int); need = set(pos) | set(pos + lag)
    cap, small, big, i = cv2.VideoCapture(f'videos/{vid}.mp4'), {}, {}, 0
    ls, lb = np.zeros((FH, FW, 3), np.uint8), np.zeros((SZ, SZ, 3), np.uint8)
    while i <= max(need):
        if i in need:
            ok, fr = cap.read()
            if ok: ls, lb = cv2.resize(fr, (FW, FH), interpolation=cv2.INTER_AREA), cv2.resize(fr, (SZ, SZ), interpolation=cv2.INTER_AREA)
            small[i], big[i] = ls, lb
        elif not cap.grab(): break
        i += 1
    feats, grays, rgb, mot = [], [], [], []
    for p in pos:
        f, g = frame_feats(small.get(p, ls), small.get(p + lag, ls)); feats.append(f); grays.append(g)
        a, b = big.get(p, lb), big.get(p + lag, lb)
        g0, g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 13, 3, 5, 1.1, 0)
        mot.append(np.stack([fl[..., 0] * 16, fl[..., 1] * 16, (g1.astype(np.float32) - g0) / 2]).clip(-127, 127).astype(np.int8))
        rgb.append(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    G = np.stack(grays).astype(np.float32); bgd = np.abs(G - np.median(G, 0))   # lệch so với nền tĩnh của video
    Fv = np.c_[np.array(feats, np.float32), bgd.mean((1, 2)), (bgd > 30).mean((1, 2)), np.r_[0, np.abs(np.diff(G, axis=0)).mean((1, 2))]]
    return Fv.astype(np.float32), np.stack(rgb), np.stack(mot), pos + 1


def load_data(vids):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if list(z['vids']) == vids: return z['F'], z['rgb'], z['mot'], z['pos']
    t0 = time.time()
    with Pool(min(8, os.cpu_count() or 4)) as p: res = p.map(read_video, vids, chunksize=2)
    Fa, rgb, mot, pos = (np.stack([r[k] for r in res]) for k in range(4))
    os.makedirs('outputs', exist_ok=True); np.savez(CACHE, F=Fa, rgb=rgb, mot=mot, pos=pos, vids=np.array(vids))
    print(f'[đọc video] {len(vids)} video, F {Fa.shape}, rgb {rgb.shape} ({time.time() - t0:.0f}s)', flush=True)
    return Fa, rgb, mot, pos


# ============================================================================ 2. mô hình bảng
def agg(F, extra):  # đặc trưng mức video góc nhìn A
    med = np.median(F, 1, keepdims=True)[:, 0]; q10, q90 = np.quantile(F, [0.1, 0.9], axis=1)
    top = np.argsort(-F[..., 0], 1)[:, :T // 4]; topm = np.take_along_axis(F, top[..., None], 1).mean(1)
    return np.c_[F.mean(1), F.std(1), F.max(1), F.min(1), q10, q90, F.max(1) - med, topm - med, np.abs(np.diff(F, axis=1)).max(1), extra]


def frame_rows(F, rel, smooth=True):  # đặc trưng theo khung: thô, chuẩn hóa robust trong video, trung bình trượt, vị trí
    med = np.median(F, 1, keepdims=True); Z = (F - med) / (np.median(np.abs(F - med), 1, keepdims=True) + 1e-4)
    parts = [F, Z]
    if smooth:
        k = np.ones(5) / 5; parts.append(np.apply_along_axis(lambda x: np.convolve(np.pad(x, 2, mode='edge'), k, 'valid'), 1, Z))
    return np.concatenate(parts + [rel[..., None]], 2)


def win_feats(F, A):  # đặc trưng cửa sổ sự kiện: F trung bình có trọng số theo xác suất khung bất thường A
    w = A / (A.sum(1, keepdims=True) + 1e-6); wm = (w[..., None] * F).sum(1); mu = F.mean(1)
    top = np.argsort(-A, 1)[:, :T // 4]; tm = np.take_along_axis(F, top[..., None], 1).mean(1)
    return np.c_[wm, wm - mu, tm - mu, A.max(1), A.mean(1), A.std(1), (A > 0.5).mean(1), np.argmax(A, 1) / T]


def fit_cls(X, y, seed):
    def proba(m, Xs):
        p = np.zeros((len(Xs), K)); p[:, m.classes_] = m.predict_proba(Xs); return p
    ms = [XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.4, tree_method='hist',
                        random_state=seed, n_jobs=8, verbosity=0).fit(X, y, sample_weight=compute_sample_weight('balanced', y)),
          make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight='balanced')).fit(X, y),
          ExtraTreesClassifier(600, max_features=0.3, class_weight='balanced_subsample', random_state=seed, n_jobs=8).fit(X, y)]
    return lambda Xs: np.mean([proba(m, Xs) for m in ms], 0)


def fit_loc(R, Yf, cls, seed):  # cls=None: không phân biệt lớp
    oh = lambda c, n: np.zeros((n, T, 0)) if cls is None else np.repeat(np.eye(K)[c][:, None], T, 1)
    flat = lambda R, c: np.concatenate([R, oh(c, len(R))], 2).reshape(len(R) * T, -1)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                      random_state=seed, n_jobs=8, verbosity=0).fit(flat(R, cls), Yf.reshape(-1))
    return lambda R2, c=None: m.predict_proba(flat(R2, c))[:, 1].reshape(len(R2), T)


class Tab:
    """Hai góc nhìn bảng A/B trên cùng đặc trưng khung."""
    def __init__(self, Fa, pos, nF, vids, y, Yf, use_win=False):
        meta = np.array([[float(INFO[v]['fps']), nF[k], int(INFO[v]['width']), int(INFO[v]['height']),
                          int(INFO[v]['width']) / int(INFO[v]['height']), nF[k] / float(INFO[v]['fps'])] for k, v in enumerate(vids)])
        rel = pos / nF[:, None]; Fb = Fa[:, :, SHORT]; medb = np.median(Fb, 1)
        self.Fa, self.y, self.Yf, self.use_win = Fa, y, Yf, use_win
        self.XA, self.RA = agg(Fa, meta), frame_rows(Fa, rel)
        self.XB = np.c_[Fb.mean(1), Fb.std(1), Fb.max(1), Fb.min(1), Fb.max(1) - medb, meta[:, :4]]
        self.RB = frame_rows(Fb, rel, smooth=False)

    def anomaly_frames(self, tr, te, seed):  # xác suất khung bất thường (không phân biệt lớp); OOF bên trong cho tập train
        y, Yf, A = self.y, self.Yf, np.zeros((len(self.RA), T))
        fit = lambda idx: fit_loc(self.RA[idx[y[idx] > 0]], Yf[idx[y[idx] > 0]], None, seed)
        for a, b in KFold(5, shuffle=True, random_state=seed).split(tr): A[tr[b]] = fit(tr[a])(self.RA[tr[b]])
        A[te] = fit(tr)(self.RA[te]); return A

    def fit(self, tr, te, seeds=(0,)):
        """-> xác suất lớp cho te (trung bình A,B) và hàm (lớp chốt) -> xác suất khung bất thường cho te."""
        y, Yf = self.y, self.Yf
        XA = np.c_[self.XA, win_feats(self.Fa, self.anomaly_frames(tr, te, seeds[0]))] if self.use_win else self.XA
        pA = np.mean([fit_cls(XA[tr], y[tr], s)(XA[te]) for s in seeds], 0)
        pB = np.mean([fit_cls(self.XB[tr], y[tr], s)(self.XB[te]) for s in seeds], 0)
        an = tr[y[tr] > 0]
        la, lb = fit_loc(self.RA[an], Yf[an], y[an], seeds[0]), fit_loc(self.RB[an], Yf[an], y[an], seeds[0])
        return pA, pB, lambda c: (la(self.RA[te], c) + lb(self.RB[te], c)) / 2


# ============================================================================ 3. CNN two-stream (random init)
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__(); self.f = nn.Sequential(nn.Linear(c, max(8, c // r)), nn.ReLU(True), nn.Linear(max(8, c // r), c))

    def forward(self, x): return x * torch.sigmoid(self.f(x.mean((2, 3))))[..., None, None]


def tsm(x, t, div=8):  # Temporal Shift Module: 1/8 kênh từ khung trước, 1/8 từ khung sau
    bt, c, h, w = x.shape; x = x.view(bt // t, t, c, h, w); f = c // div; out = torch.zeros_like(x)
    out[:, 1:, :f] = x[:, :-1, :f]; out[:, :-1, f:2 * f] = x[:, 1:, f:2 * f]; out[:, :, 2 * f:] = x[:, :, 2 * f:]
    return out.view(bt, c, h, w)


class Block(nn.Module):  # ResNet BasicBlock + SE (+ TSM)
    def __init__(self, cin, cout, stride, shift):
        super().__init__(); self.shift = shift
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout); self.se = SE(cout)
        self.down = None if stride == 1 and cin == cout else nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Identity(), nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        nn.init.zeros_(self.b2.weight)  # khối dư khởi đầu = đồng nhất

    def forward(self, x):
        h = self.se(self.b2(self.c2(F.relu(self.b1(self.c1(tsm(x, TS) if self.shift else x)), True))))
        return F.relu(h + (x if self.down is None else self.down(x)), True)


class Stream(nn.Module):
    def __init__(self, cin, stem, chans, blocks, strides):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, stem, 3, 2, 1, bias=False), nn.BatchNorm2d(stem), nn.ReLU(True),
                                  nn.Conv2d(stem, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem), nn.ReLU(True))
        layers, c = [], stem
        for s, (co, nb, st) in enumerate(zip(chans, blocks, strides)):
            for b in range(nb): layers.append(Block(c, co, st if b == 0 else 1, shift=s >= 1)); c = co
        self.layers, self.out = nn.Sequential(*layers), c

    def forward(self, x): return self.layers(self.stem(x)).mean((2, 3))


class TCN(nn.Module):
    def __init__(self, c, d):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm1d(c),
                                                     nn.GELU(), nn.Dropout(0.2), nn.Conv1d(c, c, 1))

    def forward(self, x): return x + self.net(x)


class MSADNet(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.app = Stream(3, 32, [32, 64, 128, 192], [1, 2, 2, 1], [1, 2, 2, 2])   # RGB: ngoại hình bối cảnh
        self.mot = Stream(4, 24, [24, 48, 96], [1, 1, 1], [2, 2, 2])               # flow x, y, |flow|, hiệu xám
        self.fuse = nn.Sequential(nn.Linear(self.app.out + self.mot.out, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.3))
        self.tpos = nn.Parameter(torch.zeros(1, d, TS)); self.tcn = nn.Sequential(TCN(d, 1), TCN(d, 2), TCN(d, 4))
        self.frame_head, self.att = nn.Linear(d, K), nn.Linear(d, 1)
        self.video_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(2 * d, K))

    def forward(self, rgb, mot):
        B = rgb.shape[0]; f = torch.cat([self.app(rgb.flatten(0, 1)), self.mot(mot.flatten(0, 1))], 1)
        h = self.tcn(self.fuse(f).view(B, TS, -1).transpose(1, 2) + self.tpos).transpose(1, 2)
        v = torch.cat([(self.att(h).softmax(1) * h).sum(1), h.max(1).values], 1)    # attention pooling ++ max pooling
        return self.frame_head(h), self.video_head(v)


def to_gpu(rgb, mot):
    r = (torch.from_numpy(rgb).to(DEV).permute(0, 1, 4, 2, 3).float() / 255 - 0.45) / 0.25
    m = torch.from_numpy(mot).to(DEV).float() / 32
    return r, torch.cat([m, m[:, :, :2].norm(dim=2, keepdim=True)], 2)


def flip_mot(m): return torch.cat([-m[..., :1, :, :], m[..., 1:, :, :]], -3).flip(-1)  # lật ngang: đổi dấu flow-x


def augment(r, m):  # cùng phép biến đổi cho cả clip
    outr, outm = [], []
    for i in range(len(r)):
        sc, ar = np.random.uniform(0.55, 1.0), np.random.uniform(0.8, 1.25)
        ch, cw = min(SZ, int(SZ * sc * ar ** .5)), min(SZ, int(SZ * sc / ar ** .5))
        y0, x0 = np.random.randint(0, SZ - ch + 1), np.random.randint(0, SZ - cw + 1)
        ri = F.interpolate(r[i, :, :, y0:y0 + ch, x0:x0 + cw], (OUT, OUT), mode='bilinear', align_corners=False)
        mi = F.interpolate(m[i, :, :, y0:y0 + ch, x0:x0 + cw], (OUT, OUT), mode='bilinear', align_corners=False)
        if np.random.rand() < 0.5: ri, mi = ri.flip(-1), flip_mot(mi)
        ri = ri * np.random.uniform(0.7, 1.3) + np.random.uniform(-0.4, 0.4) + torch.randn(1, 3, 1, 1, device=DEV) * 0.1
        drop = torch.rand(TS, device=DEV) < 0.1
        if drop.any():
            src = torch.arange(TS, device=DEV); src[drop] = (src[drop] - 1).clamp(min=0); ri, mi = ri[src], mi[src]
        outr.append(ri); outm.append(mi)
    return torch.stack(outr).contiguous(), torch.stack(outm).contiguous()  # non-contiguous làm cuDNN chậm ~4x


def train_cnn(rgb, mot, y, Yc, seed, epochs, bs=4, lr=2e-3, wd=0.05, ema_decay=0.995, log=''):
    seed_all(seed); model = MSADNet().to(DEV); ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    cnt = np.bincount(y, minlength=K).astype(float); sw = 1 / np.sqrt(cnt[y]); sw /= sw.sum()
    cw = lambda c: torch.tensor((c.sum() / (K * np.maximum(c, 1))) ** .5, dtype=torch.float32, device=DEV)
    cw_v, cw_f = cw(cnt), cw(np.bincount(Yc.ravel(), minlength=K).astype(float))
    dec = [p for n, p in model.named_parameters() if p.ndim > 1 and 'tpos' not in n]
    nod = [p for n, p in model.named_parameters() if not (p.ndim > 1 and 'tpos' not in n)]
    opt = torch.optim.AdamW([{'params': dec, 'weight_decay': wd}, {'params': nod, 'weight_decay': 0}], lr=lr)
    spe = math.ceil(len(y) / bs); total, warm = epochs * spe, 3 * spe
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else .5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    t0, step = time.time(), 0
    for ep in range(epochs):
        model.train(); tl = 0
        for _ in range(spe):
            idx = np.random.choice(len(y), bs, p=sw); off = np.random.randint(STRIDE)
            r, m = augment(*to_gpu(rgb[idx][:, off::STRIDE], mot[idx][:, off::STRIDE]))
            fl, vl = model(r, m)
            loss = F.cross_entropy(vl, torch.from_numpy(y[idx]).to(DEV), weight=cw_v, label_smoothing=0.1) + 0.5 * F.cross_entropy(
                fl.flatten(0, 1), torch.from_numpy(Yc[idx][:, off::STRIDE]).to(DEV).flatten(), weight=cw_f, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sch.step(); step += 1; d = min(ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                    pe.copy_(pe * d + pm * (1 - d)) if pe.dtype.is_floating_point else pe.copy_(pm)
            tl += loss.item()
        if (ep + 1) % 10 == 0: print(f'  [CNN{log}] ep {ep + 1}/{epochs} loss {tl / spe:.3f} ({time.time() - t0:.0f}s)', flush=True)
    return ema


@torch.no_grad()
def predict_cnn(model, rgb, mot, bs=4):  # 3 lượt lệch khung x 2 lật -> xác suất video và xác suất từng khung (48)
    model.eval(); P, A = [], []
    rs = lambda x: F.interpolate(x.flatten(0, 1), (OUT, OUT), mode='bilinear', align_corners=False).view(*x.shape[:3], OUT, OUT)
    for i in range(0, len(rgb), bs):
        r, m = to_gpu(rgb[i:i + bs], mot[i:i + bs]); r, m = rs(r), rs(m); ps, fs = 0, torch.zeros(len(r), T, K, device=DEV)
        for off in range(STRIDE):
            for flip in (False, True):
                rr, mm = r[:, off::STRIDE].contiguous(), m[:, off::STRIDE].contiguous()
                if flip: rr, mm = rr.flip(-1), flip_mot(mm)
                fl, vl = model(rr, mm); ps = ps + vl.softmax(-1) / (2 * STRIDE); fs[:, off::STRIDE] += fl.softmax(-1) / 2
        P.append(ps.cpu()); A.append(fs.cpu())
    return torch.cat(P).numpy(), 1 - torch.cat(A).numpy()[..., 0]   # A = P(khung bất thường)


# ============================================================================ 4. giải mã + ensemble
def segment(a, pos, n, thr):  # đoạn liên tục có tổng (p - thr) lớn nhất (Kadane), đổi ra chỉ số khung 1-based
    a = np.convolve(np.pad(a, 1, mode='edge'), [.25, .5, .25], 'valid') - thr; best, cur, s0, bs, be = -1e9, 0, 0, 0, 0
    for t, x in enumerate(a):
        cur, s0 = (x, t) if cur <= 0 else (cur + x, s0)
        if cur > best: best, bs, be = cur, s0, t
    s = 1 if bs == 0 else (pos[bs - 1] + pos[bs]) // 2 + 1; e = n if be == T - 1 else (pos[be] + pos[be + 1]) // 2
    return int(s), int(max(s, min(e, n)))


def ensemble(pA, pB, locf, pC, aC, wc, ns):
    p = (1 - wc) * (pA + pB) / 2 + wc * pC; p[:, 0] *= ns; c = p.argmax(1)
    return c, (1 - wc) * locf(c) + wc * aC


def decode(vids, c, A, pos, nF, thr):
    return {v: (L[k], *map(str, segment(a, p, n, thr))) if k else ('Normal', '', '') for v, k, a, p, n in zip(vids, c, A, pos, nF)}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv', action='store_true'); ap.add_argument('--ablate', action='store_true')
    ap.add_argument('--win', action='store_true', help='bật đặc trưng cửa sổ sự kiện (OOF: A tăng nhưng ensemble giảm)')
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--cnn_seeds', type=int, default=1)
    ap.add_argument('--out', default='submission.csv'); a = ap.parse_args()
    t0 = time.time(); seed_all(0); torch.backends.cudnn.benchmark = True
    lab = rows('train_labels.csv') + rows('validation_labels.csv'); te_ids = [r['video_id'] for r in rows('test_videos.csv')]
    vids = [r['video_id'] for r in lab] + te_ids; m = len(lab); te = np.arange(m, len(vids))
    y = np.array([L.index(r['label']) for r in lab], dtype=np.int64); se = np.array([[int(r['start_frame'] or 0), int(r['end_frame'] or 0)] for r in lab])
    nF = np.array([int(INFO[v]['frames']) for v in vids])
    Fa, rgb, mot, pos = load_data(vids)
    Yf = ((pos[:m] >= se[:, :1]) & (pos[:m] <= se[:, 1:]) & (y[:, None] > 0)).astype(int); Yc = (Yf * y[:, None]).astype(np.int64)
    print(f'[dữ liệu] {m} video có nhãn, {len(te)} test ({time.time() - t0:.0f}s)', flush=True)

    if a.cv:
        ans = {vids[i]: {'label': L[y[i]], 'frames': int(nF[i]), 'start': int(se[i, 0]) or None, 'end': int(se[i, 1]) or None} for i in range(m)}
        sc = lambda pred: scorer.SCORERS['MSAD'](vids[:m], ans, pred)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(m), y))
        if os.path.exists(CNN_OOF):
            z = np.load(CNN_OOF); cP, cA = z['P'], z['A']; print('[CNN] dùng OOF đã lưu', CNN_OOF)
        else:
            cP, cA = np.zeros((m, K)), np.zeros((m, T))
            for k, (tr, va) in enumerate(folds):
                net = train_cnn(rgb[tr], mot[tr], y[tr], Yc[tr], k, a.epochs, log=f' fold {k}'); cP[va], cA[va] = predict_cnn(net, rgb[va], mot[va])
            np.savez(CNN_OOF, P=cP, A=cA)
        for use_win in ([False, True] if a.ablate else [a.win]):
            tab = Tab(Fa, pos, nF.astype(float), vids, y, Yf, use_win); oA, oB = np.zeros((m, K)), np.zeros((m, K)); locs = []
            for tr, va in folds:
                pA, pB, lf = tab.fit(tr, va); oA[va], oB[va] = pA, pB; locs.append((va, lf))
            print(f'\n=== OOF 5-fold, đặc trưng cửa sổ sự kiện: {use_win}  ({time.time() - t0:.0f}s)')
            for name, (wa, wb, wc) in {'A': (1, 0, 0), 'B': (0, 1, 0), 'CNN': (0, 0, 1)}.items():
                p = wa * oA + wb * oB + wc * cP; p[:, 0] *= NSCALE; c = p.argmax(1); r = sc(decode(vids[:m], c, cA, pos, nF, -1))
                print(f'  chỉ {name:3s} (khoảng = cả video): score {r["score"]:.4f} f1 {r["macro_f1"]:.3f} acc {r["accuracy"]:.3f}')
            res = {}
            for wc in (0.0, 0.1, 0.2, 0.3):
                for ns in (0.8, 1.0):
                    c, A = np.zeros(m, int), np.zeros((m, T))
                    for va, lf in locs: c[va], A[va] = ensemble(oA[va], oB[va], lf, cP[va], cA[va], wc, ns)
                    for thr in (0.3, 0.4, 0.5): res[(wc, ns, thr)] = sc(decode(vids[:m], c, A, pos, nF, thr))
            for wc in (0.0, 0.1, 0.2, 0.3):
                print(f'  ensemble wc={wc}: ' + '  '.join(f'ns{ns}/thr{t} {res[(wc, ns, t)]["score"]:.4f}' for ns in (0.8, 1.0) for t in (0.3, 0.4, 0.5)))
            r = res[(W_CNN, NSCALE, THR)]
            print(f'  CẤU HÌNH NỘP (wc={W_CNN}, ns={NSCALE}, thr={THR}): score {r["score"]:.4f} f1 {r["macro_f1"]:.3f} '
                  f'acc {r["accuracy"]:.3f} mIoU {r["frame_miou"]:.3f}\n  theo lớp: {r["score_by_class"]}')
        return

    # ---------------- nộp bài: train trên toàn bộ 250 video
    tab = Tab(Fa, pos, nF.astype(float), vids, y, Yf, use_win=a.win)
    pA, pB, lf = tab.fit(np.arange(m), te, seeds=(0, 1, 2)); print(f'[bảng] xong ({time.time() - t0:.0f}s)', flush=True)
    pC, aC = 0, 0
    for s in range(a.cnn_seeds):
        net = train_cnn(rgb[:m], mot[:m], y, Yc, 100 + s, a.epochs, log=f' seed {s}'); p, q = predict_cnn(net, rgb[m:], mot[m:])
        pC, aC = pC + p / a.cnn_seeds, aC + q / a.cnn_seeds
    c, A = ensemble(pA, pB, lf, pC, aC, W_CNN, NSCALE); pred = decode(te_ids, c, A, pos[m:], nF[m:], THR)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(scorer.HEADERS['MSAD']); w.writerows([v, *pred[v]] for v in te_ids)
    err = scorer.validate('MSAD', open(a.out, 'rb').read(), te_ids, {v: {'frames': int(n)} for v, n in zip(te_ids, nF[m:])})
    print(f'[xong] {a.out}  hợp lệ: {not err} {err}  nhãn: {dict(Counter(p[0] for p in pred.values()))}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
