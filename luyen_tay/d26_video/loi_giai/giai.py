import os, random, time
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
import cv2, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8
tr, sub = pd.read_csv("data/train.csv"), pd.read_csv("data/sample_submission.csv")
NF = pd.read_csv("data/video_info.csv").set_index("video_id").frames
CLS = sorted(tr.label.unique()); NORMAL = CLS.index("binh_thuong"); N = 24

def read_video(vid):
    """N khung rải đều (đọc tuần tự 1 lượt), trả ảnh RGB (N,64,64,3) và vị trí khung 1-based."""
    pos = np.linspace(0, NF[vid] - 1, N).round().astype(int); cap, out, k = cv2.VideoCapture(f"data/videos/{vid}.mp4"), {}, 0
    while k <= pos[-1]:
        ok, f = cap.read()
        if not ok: break
        if k in set(pos): out[k] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        k += 1
    return np.stack([out.get(p, out[max(out)]) for p in pos]), pos + 1

def feats(ids, cache):
    if os.path.exists(cache): z = np.load(cache); return z["F"], z["P"]
    net = torchvision.models.resnet18(weights="DEFAULT"); net.fc = nn.Identity(); net = net.to(DEV).eval()
    mean, std = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None], torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]
    Fs, Ps = [], []
    with torch.inference_mode(), torch.autocast(DEV, torch.bfloat16, enabled=AMP):
        for vid in ids:
            im, p = read_video(vid)
            x = F.interpolate(torch.from_numpy(im).to(DEV).permute(0, 3, 1, 2).float() / 255, size=112, mode="bilinear")
            Fs.append(net((x - mean) / std).float().cpu().numpy()); Ps.append(p)
    np.savez(cache, F=np.stack(Fs), P=np.stack(Ps)); return np.stack(Fs), np.stack(Ps)

t0 = time.time()
Ftr, Ptr = feats(tr.video_id, "feat_train.npz"); Fte, Pte = feats(sub.video_id, "feat_test.npz")
print(f"đặc trưng {Ftr.shape} ({time.time() - t0:.0f}s)")
vfeat = lambda Fv: np.concatenate([Fv.mean(1), Fv.std(1), np.abs(Fv[:, 2:] - Fv[:, :-2]).mean(1)], 1)   # gộp thời gian + hiệu t/t+2
ffeat = lambda Fv: np.concatenate([Fv, Fv - np.median(Fv, 1, keepdims=True)], 2)                     # mỗi khung: gốc + lệch so với trung vị video
y = tr.label.map(CLS.index).values
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
oof = cross_val_predict(clf, vfeat(Ftr), y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print("OOF macro-F1 phân loại:", round(f1_score(y, oof, average="macro"), 4))
clf.fit(vfeat(Ftr), y)

ab = y != NORMAL
Yf = (Ptr >= tr.start_frame.values[:, None]) & (Ptr <= tr.end_frame.values[:, None])
loc = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
loc.fit(ffeat(Ftr[ab]).reshape(-1, 2 * Ftr.shape[2]), Yf[ab].ravel())                # gộp (video, khung) thành từng mẫu

def segment(p, pos, n):
    """Đoạn [i, j] tối đa hoá tIoU kỳ vọng sum_trong(p) / (độ dài + sum_ngoài(p)), có trọng số số khung mỗi ô."""
    p = np.convolve(np.pad(p, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    lo = np.r_[1, (pos[:-1] + pos[1:]) // 2 + 1]; hi = np.r_[(pos[:-1] + pos[1:]) // 2, n]
    w = hi - lo + 1; P = np.r_[0, np.cumsum(p * w)]; W = np.r_[0, np.cumsum(w)]
    i, j = np.triu_indices(len(p)); inter = P[j + 1] - P[i]
    k = (inter / (W[j + 1] - W[i] + P[-1] - inter)).argmax()
    return int(lo[i[k]]), int(hi[j[k]])

c = clf.predict(vfeat(Fte))
pf = loc.predict_proba(ffeat(Fte).reshape(-1, 2 * Fte.shape[2]))[:, 1].reshape(len(Fte), N)
rows = []
for vid, k, p, pos in zip(sub.video_id, c, pf, Pte):
    s, e = segment(p, pos, NF[vid]) if k != NORMAL else ("", "")
    rows.append((vid, CLS[k], s, e))
out = pd.DataFrame(rows, columns=["video_id", "label", "start_frame", "end_frame"])
out.to_csv("submission.csv", index=False)
print("xong", f"({time.time() - t0:.0f}s)", out.label.value_counts().to_dict())
