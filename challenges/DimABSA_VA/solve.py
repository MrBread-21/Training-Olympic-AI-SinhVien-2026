"""Lời giải DimABSA_VA (Subtask 1, không pretrained): đặc trưng của mệnh đề chứa khía cạnh + cửa sổ rộng + cả câu,
TF-IDF từ 1-3 gram và ký tự, Ridge riêng cho valence và arousal; trộn với Ridge dự đoán arousal từ |valence - 5| (quan hệ chữ V).
Chạy: python solve.py <thư_mục_dữ_liệu> (và private_test.jsonl nếu đã giải nén). Ghi pred_test.jsonl (định dạng đề) và submission.csv."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

np.random.seed(42)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
doc = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
train, dev, test = doc(D / "train.jsonl"), doc(D / "dev.jsonl"), doc(D / "test.jsonl")
for p in [Path("private_test.jsonl"), D / "private_test.jsonl"]:
    if p.exists(): test += doc(p); break
NGAT = {",", ".", ";", "!", "?", "but", "and", "though", "although", "however", "while", "-", "--", "whereas", "yet"}
def tok(s):
    """Chữ thường; gộp phủ định về một dạng: train đã tách sẵn "ca n ' t", test viết liền "can't"."""
    s = re.sub(r"\s*n\s*'\s*t\b", " not", s.lower())
    s = re.sub(r"\s*'\s*(s|re|ve|ll|d|m)\b", r" '\1", s)
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9']", s)


def ba_phan(text, asp):
    """(mệnh đề chứa khía cạnh, cửa sổ ±8 từ, cả câu); khía cạnh thay bằng ASP."""
    t, a = tok(text), tok(asp)
    for i in range(len(t) - len(a) + 1):
        if t[i:i + len(a)] == a:
            l, r = i, i + len(a)
            while l > 0 and t[l - 1] not in NGAT: l -= 1
            while r < len(t) and t[r] not in NGAT: r += 1
            moi = t[:i] + ["ASP"] + t[i + len(a):]
            return " ".join(t[l:i] + ["ASP"] + t[i + len(a):r]), " ".join(t[max(0, i - 8):i] + ["ASP"] + t[i + len(a):i + len(a) + 8]), " ".join(moi)
    s = " ".join(t)
    return s, s, s


def mau(rows, co_nhan):
    P, Y = [], []
    for r in rows:
        for a in r["aspects"]:
            P.append(ba_phan(r["text"], a["aspect"] if co_nhan else a["term"]))
            if co_nhan: Y.append((a["valence"], a["arousal"]))
    return P, np.array(Y)


vec = [(TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", sublinear_tf=True, min_df=1), 0, 1.0),
       (TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2), 0, 1.0),
       (TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\S+", sublinear_tf=True), 1, 0.6),
       (TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\S+", sublinear_tf=True), 2, 0.3)]
def X_(P, fit=False):
    return hstack([w * (v.fit_transform([p[k] for p in P]) if fit else v.transform([p[k] for p in P])) for v, k, w in vec]).tocsr()

def fit_predict(tr_rows, P_te, alpha=1.5):
    P, Y = mau(tr_rows, True); Xtr = X_(P, True); Xte = X_(P_te)
    v = Ridge(alpha=alpha).fit(Xtr, Y[:, 0]).predict(Xte)
    a1 = Ridge(alpha=alpha).fit(Xtr, Y[:, 1]).predict(Xte)
    vv = Ridge(alpha=1.0).fit(np.column_stack([np.abs(Y[:, 0] - 5), Y[:, 0]]), Y[:, 1])   # arousal theo |V - 5|
    a2 = vv.predict(np.column_stack([np.abs(v - 5), v]))
    return np.clip(np.column_stack([v, 0.6 * a1 + 0.4 * a2]), 1, 9)

Pd, Yd = mau(dev, True)
P = fit_predict(train, Pd)
print(f"dev RMSE_VA = {np.sqrt(np.mean(((P - Yd) ** 2).sum(1))):.4f}")
# mô hình co về trung bình (hệ số góc < 1): học phép giãn tuyến tính trên dev (dự đoán của mô hình chỉ train trên train)
gian = [np.polyfit(P[:, k], Yd[:, k], 1) for k in range(2)]
print("hệ số giãn (V, A):", [round(g[0], 2) for g in gian])
Pg = np.clip(np.column_stack([np.polyval(gian[k], P[:, k]) for k in range(2)]), 1, 9)
print(f"dev RMSE_VA sau giãn (lạc quan vì học trên chính dev) = {np.sqrt(np.mean(((Pg - Yd) ** 2).sum(1))):.4f}")
Pt, _ = mau(test, False); P = fit_predict(train + dev, Pt)
P = np.clip(np.column_stack([np.polyval(gian[k], P[:, k]) for k in range(2)]), 1, 9).round(2); k = 0; out = []
for r in test:
    asp = []
    for a in r["aspects"]:
        asp.append({"aspect": a["term"], "valence": float(P[k, 0]), "arousal": float(P[k, 1])}); k += 1
    out.append({"id": r["id"], "aspects": asp})
with open("pred_test.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
pd.DataFrame({"id": [o["id"] for o in out], "aspects": [json.dumps(o["aspects"], ensure_ascii=False) for o in out]}).to_csv("submission.csv", index=False)
print("đã ghi pred_test.jsonl và submission.csv:", len(out), "câu")
