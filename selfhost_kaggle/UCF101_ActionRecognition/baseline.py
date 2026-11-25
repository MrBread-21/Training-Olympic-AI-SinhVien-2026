"""Baseline video: lấy khung giữa của clip, đặc trưng ResNet-18 ImageNet + LogisticRegression.
Chạy: python baseline.py [thư_mục_dữ_liệu]  -> submission.csv"""
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

torch.manual_seed(42); np.random.seed(42)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
m = torchvision.models.resnet18(weights="IMAGENET1K_V1"); m.fc = torch.nn.Identity(); m = m.eval().to(DEV)
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def khung_giua(p):
    cap = cv2.VideoCapture(str(p)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2); ok, f = cap.read(); cap.release()
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def dac_trung(paths):
    with torch.inference_mode():
        return np.concatenate([m(tf(khung_giua(p))[None].to(DEV)).cpu().numpy() for p in paths])


tr, te = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
Xtr = dac_trung([D / f"train/{i}.avi" for i in tr.id]); Xte = dac_trung([D / f"test/{i}.avi" for i in te.id])
clf = LogisticRegression(max_iter=3000).fit(Xtr, tr.label)
pd.DataFrame({"id": te.id, "label": clf.predict(Xte)}).to_csv("submission.csv", index=False)
print("xong")
