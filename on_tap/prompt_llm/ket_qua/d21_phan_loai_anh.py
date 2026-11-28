import torch
import os, random, json, math, re, time, unicodedata, heapq
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


seed_all(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def amp():
    return torch.autocast(dev, torch.bfloat16, enabled=use_bf16)
# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/sample_submission.csv")
classes = sorted(train["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}

# Preprocess images
def load(ids, folder):
    images = []
    for i in ids:
        img = Image.open(f"{folder}/{i}.png").convert("RGB")
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        images.append(img)
    return torch.stack(images)

# Split data
X = load(train["id"], "data/train")
y = torch.tensor([class_to_idx[l] for l in train["label"]], dtype=torch.long)
train_idx, val_idx = train_test_split(np.arange(len(train)), test_size=0.15, stratify=y.numpy(), random_state=42)

# Create DataLoaders
train_dataset = TensorDataset(X[train_idx], y[train_idx])
val_dataset = TensorDataset(X[val_idx], y[val_idx])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)

# Preprocessing
def prep(x, train=True):
    x = x.to(dev).float() / 255
    if train:
        if random.random() < 0.5:
            x = torch.flip(x, [3])
        k = random.randint(0, 3)
        x = torch.rot90(x, k, [2, 3])
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return (x - mean) / std

# Model setup
model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(512, 6)
model.to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=8 * len(train_loader))
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
best_f1 = 0
best_model = None
for epoch in range(8):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        x, y = batch
        x = prep(x, train=True)
        with amp():
            outputs = model(x)
            loss = criterion(outputs, y.to(dev))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_preds = []
    val_true = []
    with torch.inference_mode():
        for batch in val_loader:
            x, y = batch
            x = prep(x, train=False)
            with amp():
                outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(y.cpu().numpy())
    val_f1 = f1_score(val_true, val_preds, average='macro')
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best.pt")

# Predict test
model.load_state_dict(torch.load("best.pt"))
test_ids = test["id"]
test_images = load(test_ids, "data/test")
test_dataset = TensorDataset(test_images)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)
model.eval()
test_preds = []
with torch.inference_mode():
    for batch in test_loader:
        x = batch[0]
        x = prep(x, train=False)
        with amp():
            outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
test_preds = [idx_to_class[p] for p in test_preds]
test["label"] = test["id"].map(dict(zip(test_ids, test_preds)))
assert test["label"].notna().all()
test.to_csv("submission.csv", index=False)
