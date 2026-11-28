import torch
import os, random, json, math, re, time, unicodedata, heapq
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.io import wavfile
from scipy.sparse import hstack
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, StratifiedGroupKFold,
                                     GroupShuffleSplit, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


seed_all(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def amp():
    return torch.autocast(dev, torch.bfloat16, enabled=use_bf16)
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/sample_submission.csv")

def feats(ids, folder):
    if os.path.exists(f"feat_{folder}.npy"):
        return np.load(f"feat_{folder}.npy")
    else:
        paths = [f"data/{folder}/{id}.png" for id in ids]
        images = []
        for i in range(0, len(paths), 64):
            batch = [np.array(Image.open(p).convert("RGB")) for p in paths[i:i+64]]
            batch = np.stack(batch).astype(np.float32)/255
            batch = np.transpose(batch, (0,3,1,2))
            batch = F.interpolate(torch.from_numpy(batch), size=(224,224), mode='bilinear', align_corners=False)
            batch = (batch - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            with torch.inference_mode(), amp():
                f = backbone(batch.to(dev)).float().cpu().numpy()
            images.append(f)
        images = np.concatenate(images)
        np.save(f"feat_{folder}.npy", images)
        return images

backbone = torchvision.models.resnet50(weights="DEFAULT").to(dev)
backbone.eval()
backbone.fc = nn.Identity()

train_feats = feats(train.id, "train")
test_feats = feats(test.id, "test")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")

for model in [lr, knn]:
    oof = cross_val_predict(model, train_feats, train.label, cv=sgkf, groups=train.group)
    print(f"Grouped CV {model.__class__.__name__} F1:", f1_score(train.label, oof, average="macro"))
    if model is lr:  # SỬA TAY: chatbot viết isinstance(model, make_pipeline)
        cm = confusion_matrix(train.label, oof, labels=sorted(train.label.unique()))
        print("Confusion Matrix:", cm)

oof = cross_val_predict(lr, train_feats, train.label, cv=skf)
print(f"Non-grouped CV LR F1:", f1_score(train.label, oof, average="macro"))

oof_df = pd.DataFrame({"id": train.id, "label": oof})
oof_df.to_csv("oof.csv", index=False)

lr.fit(train_feats, train.label)
test_pred = lr.predict(test_feats)
submission = pd.DataFrame({"id": test.id, "label": test_pred})
submission.to_csv("submission.csv", index=False)

assert not submission.isnull().values.any()
