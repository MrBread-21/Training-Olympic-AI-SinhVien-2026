# CCTV Street Scene: điểm bất thường cho từng khung (AUROC gộp mọi khung test). Chạy trên CPU, không dùng trọng số pretrained.
# điểm = z(trong video) + a * z(mức video)
#   mức video : -log(chuyển động trung bình của cả video); cảnh bất thường được dàn dựng lúc đường vắng
#   trong video: khoảng cách kNN của blob (vị trí, kích thước, chuyển động, hướng) tới blob của 46 video train,
#                trung bình 7 biến thể (nền theo video, nền LCN chung), chuẩn hoá z theo video, làm mượt sigma=3
# Chạy: python solve.py   (1-2 phút)
import os, sys, glob, time, argparse, warnings, numpy as np, pandas as pd, cv2
try: sys.stdout.reconfigure(encoding="utf-8")                 # in tiếng Việt trên console Windows
except Exception: pass
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore", category=UserWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--out", default="submission.csv")
ap.add_argument("--a", type=float, default=2)                # trọng số phần mức video
ap.add_argument("--sigma", type=float, default=3)            # làm mượt điểm trong video theo thời gian
args, _ = ap.parse_known_args()
np.random.seed(42)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

# ---------------- 0. dữ liệu ----------------
VIDS = []                                                     # (split, video, [đường dẫn ảnh], [id])
for split in ["train", "validation", "test"]:
    ds = sorted(glob.glob(os.path.join(args.root, split, "*", "")))
    assert ds, f"không thấy video trong {split}/"
    for d in ds:
        d = os.path.normpath(d); v = os.path.basename(d); fs = sorted(glob.glob(os.path.join(d, "*.jpg")))
        VIDS.append((split, v, fs, [f"{v}_{os.path.splitext(os.path.basename(f))[0]}" for f in fs]))
ids = np.array([i for *_, ii in VIDS for i in ii]); vid = np.array([v for _, v, fs, _ in VIDS for _ in fs])
spl = np.array([s for s, _, fs, _ in VIDS for _ in fs]); N = len(ids)
lab = pd.read_csv(os.path.join(args.root, "validation_labels.csv")).set_index("id")["label"]
sub = pd.read_csv(os.path.join(args.root, "sample_submission.csv"))
istr, isva, iste = spl == "train", np.isin(ids, lab.index), np.isin(ids, sub["id"])
assert isva.sum() == len(lab) and iste.sum() == len(sub), "id ảnh không khớp nhãn / file mẫu"
y = np.zeros(N); y[isva] = lab.loc[ids[isva]].values
V = list(dict.fromkeys(vid)); VA = [v for v in V if v.startswith("va")]
log(f"train {istr.sum()} khung | val {isva.sum()} khung / {len(VA)} video | test {iste.sum()} khung")

def gray(fs, size=None):
    G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]
    assert all(g is not None for g in G), "lỗi đọc ảnh"
    if size: G = [cv2.resize(g, size, interpolation=cv2.INTER_AREA) for g in G]
    return np.stack(G).astype(np.float32) / 255

# ---------------- 1. tách blob ----------------
# đặc trưng mỗi blob: cx, cy, log(diện tích), h/w, chuyển động trong khung bao, khoảng cách tới blob gần nhất ở khung trước,
#                     dx, dy (hướng dịch chuyển), tỉ lệ pixel đứng yên
COLS = {"cx": 0, "cy": 1, "la": 2, "ar": 3, "mot": 4, "disp": 5, "dx": 6, "dy": 7, "stat": 8}
def blobs_of(G, masks, min_area, mov_thr):
    rows, prev = [], np.empty((0, 2), np.float32)
    Hh, Ww = G.shape[1:]
    for t in range(len(G)):
        fr, pf = G[t], G[t - 1] if t else G[t]
        mov = (np.abs(fr - pf) > mov_thr).astype(np.uint8)
        n, lb, st, ce = cv2.connectedComponentsWithStats(masks[t], connectivity=8)
        cur = []
        for i in range(1, n):
            x, yy, w, h, area = st[i]
            if area < min_area: continue
            cx, cy = ce[i, 0] / Ww, ce[i, 1] / Hh
            motion = np.abs(fr[yy:yy + h, x:x + w] - pf[yy:yy + h, x:x + w]).mean() if t else 0.0
            static = 1 - mov[yy:yy + h, x:x + w][lb[yy:yy + h, x:x + w] == i].mean() if t else 0.5
            if len(prev):
                dd = prev - np.array([cx, cy]); j = np.argmin((dd ** 2).sum(1)); disp = float(np.sqrt((dd[j] ** 2).sum())); dx, dy = -dd[j]
            else: disp, dx, dy = 1.0, 0.0, 0.0
            rows.append([t, cx, cy, np.log(area + 1), h / w, motion, disp, dx, dy, static]); cur.append((cx, cy))
        prev = np.array(cur, np.float32).reshape(-1, 2)
    return rows

k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
def blobs_video_bg():                                         # nền = trung vị của chính video đó, ảnh 320x180
    out, off, mo = [], 0, np.zeros(N, np.float32)
    for _, v, fs, _ in VIDS:
        G = gray(fs); bg = np.median(G, 0)
        masks = [cv2.dilate(cv2.morphologyEx((np.abs(g - bg) > 0.08).astype(np.uint8), cv2.MORPH_OPEN, k3), k5) for g in G]
        out += [[r[0] + off] + r[1:] for r in blobs_of(G, masks, 20, 0.08)]
        g8 = np.stack([cv2.resize(g, (80, 45), interpolation=cv2.INTER_AREA) for g in G])     # cho phần mức video
        mo[off:off + len(G)] = np.abs(g8 - np.concatenate([g8[:1], g8[:-1]])).mean((1, 2))
        off += len(G)
    return np.array(out, np.float32), mo

def lcn(g, s=2):                                              # chuẩn hoá tương phản cục bộ, giảm ảnh hưởng của ánh sáng
    m = cv2.GaussianBlur(g, (0, 0), s); var = cv2.GaussianBlur((g - m) ** 2, (0, 0), s)
    return (g - m) / np.sqrt(var + 0.02 ** 2)
def blobs_lcn_bg(q=80):                                       # nền LCN chung của train, ngưỡng riêng từng pixel, ảnh 160x90
    trL = np.concatenate([np.stack([lcn(g) for g in gray(fs[::2], (160, 90))]) for s, _, fs, _ in VIDS if s == "train"])
    bg = np.median(trL, 0); thr = np.percentile(np.abs(trL - bg), q, axis=0); del trL
    out, off = [], 0
    for _, v, fs, _ in VIDS:
        G = gray(fs, (160, 90))
        masks = [cv2.dilate(cv2.morphologyEx((np.abs(lcn(g) - bg) > thr).astype(np.uint8), cv2.MORPH_OPEN, k3), k3, iterations=2) for g in G]
        out += [[r[0] + off] + r[1:] for r in blobs_of(G, masks, 8, 0.06)]
        off += len(G)
    return np.array(out, np.float32)

BV, mo = blobs_video_bg(); log("blob nền theo video:", len(BV))
BL = blobs_lcn_bg(); log("blob nền LCN chung:", len(BL))

# ---------------- 2. điểm kNN của blob so với blob train ----------------
def knn_score(B, cs, posw=3.0, k=5, agg="max"):
    fi = B[:, 0].astype(int); trb = istr[fi]
    X = B[:, [1 + COLS[c] for c in cs]].astype(np.float64)
    Z = (X - X[trb].mean(0)) / (X[trb].std(0) + 1e-6)
    Z[:, [cs.index("cx"), cs.index("cy")]] *= posw           # vị trí quan trọng hơn (làn xe, vỉa hè)
    dist = NearestNeighbors(n_neighbors=k).fit(Z[trb]).kneighbors(Z[~trb])[0].mean(1)
    fs = np.zeros(N)                                          # khung không có blob được điểm 0
    (np.maximum if agg == "max" else np.add).at(fs, fi[~trb], dist)
    return fs

def vz(s, sigma=0):                                           # (làm mượt) + z trong từng video
    out = np.zeros(N)
    for v in V:
        m = vid == v; q = gaussian_filter1d(s[m], sigma) if sigma else s[m]; out[m] = (q - q.mean()) / (q.std() + 1e-6)
    return out

base = ["cx", "cy", "la", "ar", "mot", "disp"]
VARIANTS = [("nền video", BV, dict(cs=base)), ("nền video, vị trí x5", BV, dict(cs=base, posw=5)),
            ("nền video, tổng", BV, dict(cs=base, agg="sum")), ("nền video, +dx dy", BV, dict(cs=base + ["dx", "dy"])),
            ("nền video, +dx dy stat", BV, dict(cs=base + ["dx", "dy", "stat"])),
            ("nền LCN, +dx dy stat", BL, dict(cs=base + ["dx", "dy", "stat"])), ("nền LCN, tổng", BL, dict(cs=base, agg="sum"))]
def within_auc(s): return [roc_auc_score(y[vid == v], s[vid == v]) for v in VA]
parts = []
for name, B, kw in VARIANTS:
    s = vz(knn_score(B, **kw)); parts.append(s)
    log(f"  {name:24s} val trong video {np.mean(within_auc(vz(s, args.sigma))):.4f}")
within = vz(np.mean(parts, 0), args.sigma)

# (1) mức video: z qua tất cả video val+test (không dùng nhãn)
vlog = {v: np.log(mo[vid == v].mean() + 1e-6) for v in V if not v.startswith("tr")}
arr = np.array(list(vlog.values()))
vterm = np.array([-(vlog[v] - arr.mean()) / (arr.std() + 1e-6) if v in vlog else 0.0 for v in vid])

# ---------------- 3. đánh giá val, ghi file nộp ----------------
score = within + args.a * vterm
wv = within_auc(within)
print(f"\nval: trong video TB {np.mean(wv):.4f} {np.round(wv, 2)} | chỉ mức video gộp {roc_auc_score(y[isva], vterm[isva]):.4f} "
      f"| gộp {roc_auc_score(y[isva], score[isva]):.4f} | AP {average_precision_score(y[isva], score[isva]):.4f}")
st = score[iste]
assert np.isfinite(st).all() and st.std() > 0, "điểm test lỗi (NaN / hằng số)"
sub["score"] = sub["id"].map(pd.Series(st, index=ids[iste]))
assert sub["score"].notna().all() and np.isfinite(sub["score"]).all(), "thiếu id hoặc điểm không hợp lệ"
sub.to_csv(args.out, index=False)
log("đã ghi", args.out, len(sub), "dòng")
