import json, random, re, unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
seed_all(42)
TEEN = {"ko": "không", "k": "không", "qá": "quá", "wá": "quá", "dc": "được", "đc": "được"}

NGAT = {",", ".", "!", ":", "nhưng", "mà", "và", "còn"}          # ranh giới mệnh đề

def chuan(s):
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join(TEEN.get(w, w) for w in re.findall(r"\w+|[^\w\s]", s))

def cua_so(text, asp):
    """Mệnh đề chứa khía cạnh (đã chuẩn hoá teencode); khía cạnh thay bằng ASP để mô hình không học thuộc tên món.
    Lấy cả cửa sổ ±4 từ thì lẫn opinion của mệnh đề bên cạnh: RMSE_VA 1.15 thay vì khoảng 0.4."""
    t, a = chuan(text).split(), chuan(asp).split()
    for i in range(len(t) - len(a) + 1):
        if t[i:i + len(a)] == a:
            l, r = i, i + len(a)
            while l > 0 and t[l - 1] not in NGAT: l -= 1
            while r < len(t) and t[r] not in NGAT: r += 1
            return " ".join(t[l:i] + ["ASP"] + t[i + len(a):r])
    return " ".join(t)

doc = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
tr, te = doc("data/a_train.jsonl"), doc("data/a_test.jsonl")
X_tr = [cua_so(r["text"], a["aspect"]) for r in tr for a in r["aspects"]]
Y = np.array([(a["valence"], a["arousal"]) for r in tr for a in r["aspects"]])
w_vec = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", sublinear_tf=True)    # bigram bắt "không ngon", "ngon xỉu"
c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
feats = lambda X, fit=False: hstack([w_vec.fit_transform(X), c_vec.fit_transform(X)]) if fit else hstack([w_vec.transform(X), c_vec.transform(X)])

n = int(0.8 * len(X_tr))                                      # tự chấm trên 20% cuối
Xa = feats(X_tr[:n], fit=True); m = [Ridge(alpha=0.1).fit(Xa, Y[:n, k]) for k in range(2)]
P = np.clip(np.column_stack([mk.predict(feats(X_tr[n:])) for mk in m]), 1, 9)
print(f"val RMSE_VA = {np.sqrt(np.mean(((P - Y[n:]) ** 2).sum(1))):.4f}  (đoán trung bình: {np.sqrt(np.mean(((Y[:n].mean(0) - Y[n:]) ** 2).sum(1))):.4f})")

Xa = feats(X_tr, fit=True); m = [Ridge(alpha=0.1).fit(Xa, Y[:, k]) for k in range(2)]   # bản cuối: toàn bộ train
with open("pred_a.jsonl", "w", encoding="utf-8") as f:
    for r in te:
        X = feats([cua_so(r["text"], a["term"]) for a in r["aspects"]])
        P = np.clip(np.column_stack([mk.predict(X) for mk in m]), 1, 9).round(2)
        f.write(json.dumps({"id": r["id"], "aspects": [{"aspect": a["term"], "valence": float(v), "arousal": float(ar)}
                                                       for a, (v, ar) in zip(r["aspects"], P)]}, ensure_ascii=False) + "\n")
print("đã ghi pred_a.jsonl")
