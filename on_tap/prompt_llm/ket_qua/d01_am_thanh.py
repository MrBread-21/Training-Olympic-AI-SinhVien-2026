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
test = pd.read_csv("data/test.csv")
classes = sorted(train.label.unique())
wav_tr = []
wav_te = []
y_tr = []
spk = []

def load(path):
    rate, data = wavfile.read(path)  # SỬA TAY: chatbot viết scipy.io.wavfile.read
    if len(data) < 16000:
        data = np.pad(data, (0, 16000 - len(data)), 'constant')
    else:
        data = data[:16000]
    return data.astype(np.float32) / 32768.0

for _, row in train.iterrows():
    id = row['id']
    path = f"data/train/{id}.wav"
    wav = load(path)
    wav_tr.append(wav)
    y_tr.append(classes.index(row['label']))
    spk.append(row['speaker'])

for _, row in test.iterrows():
    id = row['id']
    path = f"data/test/{id}.wav"
    wav = load(path)
    wav_te.append(wav)

wav_tr = torch.tensor(wav_tr, device=dev)
wav_te = torch.tensor(wav_te, device=dev)
y_tr = torch.tensor(y_tr, device=dev)
spk = np.array(spk)

def add_noise(w, snr_db):
    B = w.shape[0]
    noise = torch.zeros(B, 16000, device=dev)
    half = B // 2
    noise[:half] = torch.randn(half, 16000, device=dev)
    noise[half:] = torch.stack([wav_tr[random.randint(0, len(wav_tr)-1)] for _ in range(half)])
    signal_power = torch.mean(w.pow(2), dim=1) + 1e-8
    noise_power = torch.mean(noise.pow(2), dim=1) + 1e-8
    snr = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (snr * noise_power))
    noise = noise * scale.unsqueeze(1)
    return w + noise

def augment(w):
    shift = random.randint(-1600, 1600)
    w = torch.roll(w, shift, dims=1)
    gain = torch.rand(w.shape[0], 1, device=dev) * 0.8 + 0.6
    w = w * gain
    snr = torch.rand(w.shape[0], device=dev) * 20
    w = add_noise(w, snr)
    return w

window = torch.hann_window(400, device=dev)
fb = torch.zeros(64, 257, device=dev)
for i in range(64):
    f = 700 * (10 ** (i / 2595) - 1)
    if i > 0:
        fb[i-1, i] = 1
    if i < 63:
        fb[i, i] = 1
        fb[i+1, i] = 1

def logmel(w):
    spec = torch.stft(w, n_fft=512, hop_length=160, win_length=400, window=window, return_complex=True).abs() ** 2
    mel = torch.matmul(fb, spec)
    return torch.log(mel + 1e-6).unsqueeze(1)

def spec_aug(m):
    m = m.clone()
    b, _, h, w = m.shape
    for i in range(b):
        mask_len = random.randint(1, 8)
        start = random.randint(0, h - mask_len)
        m[i, :, start:start+mask_len, :] = 0
        mask_len = random.randint(1, 20)
        start = random.randint(0, w - mask_len)
        m[i, :, :, start:start+mask_len] = 0
    return m
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        layers+=[nn.BatchNorm2d(1)]
        for a,b in [(1,32),(32,64),(64,128),(128,128)]:
            layers+=[nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2)]
        layers+=[nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,9)]
        self.f=nn.Sequential(*layers)
    def forward(self,x):
        return self.f(x)
        
tr,va=next(GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=42).split(np.zeros(len(spk)),groups=spk))
torch.manual_seed(0)  # SỬA TAY: chatbot viết with torch.manual_seed(0):
va_wav=add_noise(wav_tr[va],torch.full((len(va),),10.0,device=dev))
    
model=Net().to(dev)
opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-3)  # SỬA TAY: thiếu torch.optim.
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=25*(len(tr)//64+1))  # SỬA TAY: thiếu torch.optim.lr_scheduler.
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    
def predict(w):
    model.eval()
    with torch.inference_mode():
        preds=[]
        for i in range(0,len(w),64):
            batch=w[i:i+64]
            with amp():
                logits=model(logmel(batch))
            preds.append(logits.argmax(1).cpu())
        return torch.cat(preds).numpy()
    
best_f1=0
for epoch in range(25):
    model.train()
    perm=torch.randperm(len(tr))
    loss_total=0
    for k in range(0,len(tr),64):
        idx=tr[perm[k:k+64].numpy()]
        x=spec_aug(logmel(augment(wav_tr[idx])))
        yb=y_tr[idx]
        with amp():
            logits=model(x)
            loss=criterion(logits,yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_total+=loss.item()
        sched.step()
    loss_mean=loss_total/(len(tr)//64)
    preds=predict(va_wav)
    f1=f1_score(y_tr[va].cpu(),preds,average='macro')
    print(f'Epoch {epoch+1}: loss={loss_mean:.4f}, F1={f1:.4f}')
    if f1>best_f1:
        best_f1=f1
        torch.save(model.state_dict(),'best.pt')
        
model.load_state_dict(torch.load('best.pt'))
te_wav=wav_te
te_preds=predict(te_wav)
submission=pd.DataFrame({'id':test.id,'label':[classes[i] for i in te_preds]})
submission.to_csv('submission.csv',index=False)
assert len(submission)==900
