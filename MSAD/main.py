"""MSAD: phân loại sự kiện bất thường trong video và định vị đoạn khung xảy ra sự kiện.
Backbone ImageNet của torchvision (convnext_tiny, resnet50) chỉ dùng đóng băng để trích đặc trưng;
mọi phần còn lại train trên dữ liệu của bài, seed cố định.

  python main.py                   ghi submission.csv (khoảng 16 phút trên GTX 1650, gồm cả đọc video)
  python main.py --cv [--reps 3]   điểm OOF 5-fold của từng thành phần
  python main.py --weights_dir D   nạp trọng số backbone từ D/{convnext_tiny,resnet50}.pth khi máy không có mạng

Pipeline
 1. Đọc mỗi video một lần tại 48 vị trí lấy đều:
      - cặp khung (t, t + 0.2 giây) ở 128x72: 55 đặc trưng thủ công (optical flow, hướng, lưới 3x3, divergence, màu HSV, nền)
      - khung t giữ tỉ lệ 16:9: GAP bản đồ đặc trưng cuối của convnext_tiny ở 640x360 và resnet50 ở 448x252
 2. Phân loại = 0.5 * LogReg(mean/max/std theo thời gian) + 0.5 * đầu TCN (5 seed), rồi nhân P(Normal) với 0.8.
 3. Định vị = XGBoost theo khung có điều kiện lớp trên đặc trưng thủ công, chọn đoạn tối đa hoá tIoU kỳ vọng.
OOF 5-fold x 3 lần trên 250 video: 0.335 (CV theo nhóm cảnh: 0.326). Lịch sử thử nghiệm: xem README.md.
"""
import argparse, csv, math, os, random, sys, time
from collections import Counter
from multiprocessing import Pool
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import scorer  # noqa: E402
sys.stdout.reconfigure(encoding='utf-8')

L = list(scorer.MSAD_LABELS); K = len(L)                # Normal = lớp 0
T, FW, FH = 48, 128, 72                     # số vị trí khung; khung đặc trưng thủ công
BB_SIZE = {'convnext_tiny': (640, 360), 'resnet50': (448, 252)}   # khung backbone (W, H) giữ tỷ lệ 16:9, mỗi backbone 1 cỡ
BACKBONES = tuple(BB_SIZE)
SIZES = sorted(set(BB_SIZE.values()))
W_HEAD, LR_C, N_HEADS, NSCALE, THR = 0.5, 0.0005, 5, 0.8, None  # chọn bằng OOF 5-fold x 3 rep; THR None = giải mã tIoU kỳ vọng
CACHE = 'outputs/main_cache_mr.npz'
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
    cap, small, big, i = cv2.VideoCapture(f'videos/{vid}.mp4'), {}, {s: {} for s in SIZES}, 0
    ls, lb = np.zeros((FH, FW, 3), np.uint8), {s: np.zeros((s[1], s[0], 3), np.uint8) for s in SIZES}
    pset = set(pos)
    while i <= max(need):
        if i in need:
            ok, fr = cap.read()
            if ok:
                ls = cv2.resize(fr, (FW, FH), interpolation=cv2.INTER_AREA)
                if i in pset: lb = {s: cv2.cvtColor(cv2.resize(fr, s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for s in SIZES}
            small[i] = ls
            if i in pset:
                for s in SIZES: big[s][i] = lb[s]
        elif not cap.grab(): break
        i += 1
    feats, grays = zip(*[frame_feats(small.get(p, ls), small.get(p + lag, ls)) for p in pos])
    G = np.stack(grays).astype(np.float32); bgd = np.abs(G - np.median(G, 0))      # lệch so với nền tĩnh
    Fv = np.c_[np.array(feats, np.float32), bgd.mean((1, 2)), (bgd > 30).mean((1, 2)), np.r_[0, np.abs(np.diff(G, axis=0)).mean((1, 2))]]
    return Fv.astype(np.float32), {s: np.stack([big[s].get(p, lb[s]) for p in pos]) for s in SIZES}, pos + 1


def load_backbone(name, weights_dir):
    """Trả hàm biến ảnh (B,3,H,W) thành vector GAP của bản đồ đặc trưng cuối (bỏ lớp phân loại)."""
    net = getattr(torchvision.models, name)(weights=None if weights_dir else 'DEFAULT')
    if weights_dir: net.load_state_dict(torch.load(os.path.join(weights_dir, f'{name}.pth'), map_location='cpu'))
    net = net.to(DEV).eval()
    if name.startswith('resnet'):
        body = nn.Sequential(*list(net.children())[:-2]); return lambda x: body(x).mean((2, 3))
    return lambda x: net.features(x).mean((2, 3))                                  # convnext


def load_data(vids, weights_dir=None):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if list(z['vids']) == vids: return z['F'], z['pos'], [z[n].astype(np.float32) for n in BACKBONES]
    t0 = time.time(); nets = [load_backbone(n, weights_dir) for n in BACKBONES]
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    Fs, Ps, Es = [], [], [[] for _ in BACKBONES]
    with Pool(min(6, os.cpu_count() or 4)) as pool, torch.no_grad():
        for k, (Fv, frs, p) in enumerate(pool.imap(read_video, vids, chunksize=1)):  # CPU đọc video, GPU chạy backbone
            for j, (n, net) in enumerate(zip(BACKBONES, nets)):
                x = ((torch.from_numpy(frs[BB_SIZE[n]]).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std).contiguous()
                e = torch.cat([net(h) for h in x.split(12)])                            # lô 12 khung cho vừa 4 GB VRAM
                Es[j].append(e.cpu().numpy().astype(np.float16))
            Fs.append(Fv); Ps.append(p)
            if k % 60 == 0: print(f'  [đọc video] {k}/{len(vids)} ({time.time() - t0:.0f}s)', flush=True)
    Fa, pos, Es = np.stack(Fs), np.stack(Ps), [np.stack(e) for e in Es]
    os.makedirs('outputs', exist_ok=True)
    np.savez(CACHE, F=Fa, pos=pos, vids=np.array(vids), **{n: e for n, e in zip(BACKBONES, Es)})
    print(f'[đọc video] xong {Fa.shape} {[e.shape for e in Es]} ({time.time() - t0:.0f}s)', flush=True)
    return Fa, pos, [e.astype(np.float32) for e in Es]


# ============================================================================ 2. phân loại: LR + đầu TCN trên backbone
def fit_lr(V, y):
    mm = make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=5000, class_weight='balanced')).fit(V, y)
    def proba(Vs):
        p = np.zeros((len(Vs), K)); p[:, mm.classes_] = mm.predict_proba(Vs); return p
    return proba


class TCN(nn.Module):
    def __init__(self, c, d):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm1d(c),
                                                     nn.GELU(), nn.Dropout(0.2), nn.Conv1d(c, c, 1))

    def forward(self, x): return x + self.net(x)


class Head(nn.Module):
    def __init__(self, din, d=256):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.5), nn.Linear(din, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.3))
        self.tcn = nn.Sequential(TCN(d, 1), TCN(d, 2), TCN(d, 4))
        self.fh, self.att = nn.Linear(d, K), nn.Linear(d, 1); self.vh = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, K))

    def forward(self, x):
        h = self.tcn(self.inp(x).transpose(1, 2)).transpose(1, 2)
        return self.fh(h), self.vh(torch.cat([(self.att(h).softmax(1) * h).sum(1), h.max(1).values], 1))


def train_head(X, y, Yc, seed, epochs=80, bs=16, lr=1e-3):
    """X (n,T,D) chuẩn hóa; y nhãn video; Yc nhãn khung (lớp trong khoảng, Normal ngoài). Trả hàm dự đoán xác suất video."""
    seed_all(seed); net = Head(X.shape[-1]).to(DEV)
    cnt = np.bincount(y, minlength=K).astype(float); sw = 1 / np.sqrt(cnt[y]); sw /= sw.sum()
    cw = lambda c: torch.tensor((c.sum() / (K * np.maximum(c, 1))) ** .5, dtype=torch.float32, device=DEV)
    cwv, cwf = cw(cnt), cw(np.bincount(Yc.ravel(), minlength=K).astype(float))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05); tot = epochs * math.ceil(len(y) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=tot, pct_start=0.1)
    Xt, Yt, Yct = torch.from_numpy(X).to(DEV), torch.from_numpy(y).to(DEV), torch.from_numpy(Yc).to(DEV)
    net.train()
    for _ in range(tot):
        i = torch.from_numpy(np.random.choice(len(y), bs, p=sw)).to(DEV)
        fl, vl = net(Xt[i] + 0.1 * torch.randn_like(Xt[i]))                       # nhiễu đặc trưng
        loss = F.cross_entropy(vl, Yt[i], weight=cwv, label_smoothing=0.1) \
             + 0.5 * F.cross_entropy(fl.flatten(0, 1), Yct[i].flatten(), weight=cwf, label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()

    @torch.no_grad()
    def proba(Xs): return net(torch.from_numpy(Xs).to(DEV))[1].softmax(-1).cpu().numpy()
    return proba


# ============================================================================ 3. định vị: XGB theo khung có điều kiện lớp
def frame_rows(F, rel, smooth=True):
    med = np.median(F, 1, keepdims=True); Z = (F - med) / (np.median(np.abs(F - med), 1, keepdims=True) + 1e-4)
    parts = [F, Z]
    if smooth:
        k = np.ones(5) / 5; parts.append(np.apply_along_axis(lambda x: np.convolve(np.pad(x, 2, mode='edge'), k, 'valid'), 1, Z))
    return np.concatenate(parts + [rel[..., None]], 2)


def fit_loc(R, Yf, cls, seed=0):
    flat = lambda R, c: np.concatenate([R, np.repeat(np.eye(K)[c][:, None], T, 1)], 2).reshape(len(R) * T, -1)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                      random_state=seed, n_jobs=8, verbosity=0).fit(flat(R, cls), Yf.reshape(-1))
    return lambda R2, c: m.predict_proba(flat(R2, c))[:, 1].reshape(len(R2), T)


def segment(a, pos, n, thr=None):
    """Đoạn khung 1-based từ xác suất khung a (T vị trí). thr=None: đoạn [i,j] tối đa hóa tIoU kỳ vọng
    sum_S p / (|S| + sum_ngoài p), trọng số theo số khung mỗi ô lưới. thr số: Kadane trên (p - thr) (cách cũ)."""
    a = np.convolve(np.pad(a, 1, mode='edge'), [.25, .5, .25], 'valid')
    lo = np.r_[1, (pos[:-1] + pos[1:]) // 2 + 1]; hi = np.maximum(np.r_[(pos[:-1] + pos[1:]) // 2, n], lo)   # ô t phủ khung [lo, hi]
    if thr is None:
        w = (hi - lo + 1).astype(float); P = np.r_[0, np.cumsum(np.clip(a, 1e-4, 1) * w)]; W = np.r_[0, np.cumsum(w)]
        i, j = np.triu_indices(T); inter = P[j + 1] - P[i]
        k = (inter / (W[j + 1] - W[i] + P[-1] - inter)).argmax(); bs, be = i[k], j[k]
    else:
        a = a - thr; best, cur, s0, bs, be = -1e9, 0, 0, 0, 0
        for t, x in enumerate(a):
            cur, s0 = (x, t) if cur <= 0 else (cur + x, s0)
            if cur > best: best, bs, be = cur, s0, t
    return int(lo[bs]), int(max(lo[bs], min(hi[be], n)))


def decode(vids, c, A, pos, nF, thr=None):
    return {v: (L[k], *map(str, segment(a, p, n, thr))) if k else ('Normal', '', '') for v, k, a, p, n in zip(vids, c, A, pos, nF)}


# ============================================================================ 4. mô hình đầy đủ
class Model:
    def __init__(self, Fa, pos, nF, Es, y, Yf):
        rel = pos / nF[:, None]
        self.RA, self.RB = frame_rows(Fa, rel), frame_rows(Fa[:, :, SHORT], rel, smooth=False)
        self.V = np.concatenate([np.c_[E.mean(1), E.max(1), E.std(1)] for E in Es], 1)
        X = np.concatenate(Es, 2); mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-6
        self.X = ((X - mu) / sd).astype(np.float32)               # chuẩn hóa theo mọi khung (không dùng nhãn)
        self.y, self.Yf, self.Yc = y, Yf, (Yf * y[:, None]).astype(np.int64)

    def fit_predict(self, tr, te, n_heads=N_HEADS):
        """Xác suất lớp của LR và TCN cho te, và hàm nhận lớp đã chốt, trả xác suất khung bất thường cho te."""
        y = self.y
        pl = fit_lr(self.V[tr], y[tr])(self.V[te])
        ph = np.mean([train_head(self.X[tr], y[tr], self.Yc[tr], s)(self.X[te]) for s in range(n_heads)], 0)
        an = tr[y[tr] > 0]
        la, lb = fit_loc(self.RA[an], self.Yf[an], y[an]), fit_loc(self.RB[an], self.Yf[an], y[an])
        return pl, ph, lambda c: (la(self.RA[te], c) + lb(self.RB[te], c)) / 2


def combine(pl, ph, locf, w_head=W_HEAD, ns=NSCALE):
    p = (1 - w_head) * pl + w_head * ph; p[:, 0] *= ns; c = p.argmax(1)
    return c, locf(c)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv', action='store_true'); ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--weights_dir', default=None); ap.add_argument('--out', default='submission.csv'); a = ap.parse_args()
    t0 = time.time(); seed_all(0); torch.backends.cudnn.benchmark = True
    lab = rows('train_labels.csv') + rows('validation_labels.csv'); te_ids = [r['video_id'] for r in rows('test_videos.csv')]
    vids = [r['video_id'] for r in lab] + te_ids; m = len(lab); te = np.arange(m, len(vids))
    y = np.array([L.index(r['label']) for r in lab], dtype=np.int64)
    se = np.array([[int(r['start_frame'] or 0), int(r['end_frame'] or 0)] for r in lab])
    nF = np.array([int(INFO[v]['frames']) for v in vids])
    Fa, pos, Es = load_data(vids, a.weights_dir)
    Yf = ((pos[:m] >= se[:, :1]) & (pos[:m] <= se[:, 1:]) & (y[:, None] > 0)).astype(int)
    model = Model(Fa, pos, nF.astype(float), Es, y, Yf)
    print(f'[dữ liệu] {m} video có nhãn, {len(te)} test, V {model.V.shape}, X {model.X.shape} ({time.time() - t0:.0f}s)', flush=True)

    if a.cv:
        ans = {vids[i]: {'label': L[y[i]], 'frames': int(nF[i]), 'start': int(se[i, 0]) or None, 'end': int(se[i, 1]) or None} for i in range(m)}
        sc = lambda pred: scorer.SCORERS['MSAD'](vids[:m], ans, pred)
        res = {}
        for r in range(a.reps):
            oL, oH, locs = np.zeros((m, K)), np.zeros((m, K)), []
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(m), y):
                pl, ph, lf = model.fit_predict(tr, va); oL[va], oH[va] = pl, ph; locs.append((va, lf))
            for name, w in {'chỉ LR': 0.0, 'chỉ TCN': 1.0, 'NỘP': W_HEAD}.items():
                c, A = np.zeros(m, int), np.zeros((m, T))
                for va, lf in locs: c[va], A[va] = combine(oL[va], oH[va], lf, w)
                res.setdefault(name, []).append(sc(decode(vids[:m], c, A, pos, nF, THR)))
            print(f'  rep {r}: ' + ', '.join(f'{k} {v[-1]["score"]:.4f}' for k, v in res.items()) + f' ({time.time() - t0:.0f}s)', flush=True)
        for k, v in res.items():
            print(f'{k:8s} score {np.mean([x["score"] for x in v]):.4f} ±{np.std([x["score"] for x in v]):.4f}  '
                  f'f1 {np.mean([x["macro_f1"] for x in v]):.3f}  acc {np.mean([x["accuracy"] for x in v]):.3f}  mIoU {np.mean([x["frame_miou"] for x in v]):.3f}')
        print('theo lớp (NỘP):', {c: round(np.mean([x['score_by_class'][c] for x in res['NỘP']]), 3) for c in res['NỘP'][0]['score_by_class']})
        return

    # ---------------- nộp bài: train trên toàn bộ 250 video
    pl, ph, lf = model.fit_predict(np.arange(m), te); c, A = combine(pl, ph, lf)
    pred = decode(te_ids, c, A, pos[m:], nF[m:], THR)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(scorer.HEADERS['MSAD']); w.writerows([v, *pred[v]] for v in te_ids)
    err = scorer.validate('MSAD', open(a.out, 'rb').read(), te_ids, {v: {'frames': int(n)} for v, n in zip(te_ids, nF[m:])})
    print(f'[xong] {a.out}  hợp lệ: {not err} {err}  nhãn: {dict(Counter(p[0] for p in pred.values()))}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
