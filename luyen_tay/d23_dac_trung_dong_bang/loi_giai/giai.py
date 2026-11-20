import os, random, time
import torch, torch.nn as nn, torchvision
import cv2, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8
tr, sub = pd.read_csv("data/train.csv"), pd.read_csv("data/sample_submission.csv")
CLS = sorted(tr.label.unique()); y = tr.label.map(CLS.index).values

def feats(folder, ids, cache):
    """GAP của ResNet-50 ImageNet đóng băng (ảnh gốc + lật), lưu cache .npy."""
    if os.path.exists(cache): return np.load(cache)
    net = torchvision.models.resnet50(weights="DEFAULT"); net.fc = nn.Identity(); net = net.to(DEV).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]
    out = []
    with torch.inference_mode(), torch.autocast(DEV, torch.bfloat16, enabled=AMP):
        for i in range(0, len(ids), 128):
            x = np.stack([cv2.cvtColor(cv2.imread(f"data/{folder}/{j}.png"), cv2.COLOR_BGR2RGB) for j in ids[i:i + 128]])
            x = torch.from_numpy(x).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = (torch.nn.functional.interpolate(x, size=128, mode="bilinear") - mean) / std
            out.append(((net(x) + net(x.flip(3))) / 2).float().cpu())
    F = torch.cat(out).numpy(); np.save(cache, F); return F

t0 = time.time()
Ftr, Fte = feats("train", list(tr.id), "feat_train.npy"), feats("test", list(sub.id), "feat_test.npy")
print(f"đặc trưng {Ftr.shape} {Fte.shape} ({time.time() - t0:.0f}s)")

MODELS = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
          "kNN cosine": lambda: KNeighborsClassifier(5, metric="cosine")}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)          # chia theo nhóm: ảnh cùng phiên không lọt sang val
oof = {k: np.zeros(len(y), int) for k in MODELS}
for f, (a, b) in enumerate(cv.split(Ftr, y, tr.group)):
    for k, mk in MODELS.items():
        oof[k][b] = mk().fit(Ftr[a], y[a]).predict(Ftr[b])
for k in MODELS:
    print(f"OOF macro-F1 {k}: {f1_score(y, oof[k], average='macro'):.4f}")
best = max(MODELS, key=lambda k: f1_score(y, oof[k], average="macro"))
print("chọn", best, "\nma trận nhầm lẫn OOF (hàng = thật):", CLS)
print(confusion_matrix(y, oof[best]))
pd.DataFrame({"id": tr.id, "label": [CLS[k] for k in oof[best]]}).to_csv("oof.csv", index=False)

clf = MODELS[best]().fit(Ftr, y)                                       # bản cuối: train trên toàn bộ dữ liệu có nhãn
sub["label"] = sub.id.map(dict(zip(sub.id, [CLS[k] for k in clf.predict(Fte)])))
assert sub.label.notna().all()
sub.to_csv("submission.csv", index=False)
print(f"xong ({time.time() - t0:.0f}s)")
