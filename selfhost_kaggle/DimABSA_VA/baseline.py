# %% [markdown]
# # Baseline - DimABSA Subtask 1: dự đoán Valence-Arousal cho từng khía cạnh
#
# Không dùng model pretrained: lấy cửa sổ 6 từ quanh khía cạnh, TF-IDF (từ đơn và cặp từ) rồi hồi quy Ridge
# riêng cho valence và arousal. Tự chấm RMSE_VA trên dev, rồi ghi `pred_test.jsonl` (đúng định dạng đề)
# và `submission.csv` (định dạng nộp Kaggle: `id,aspects`, cột aspects là chuỗi JSON).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

np.random.seed(42)
if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("dev.jsonl")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)


def doc(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cua_so(text, asp, w=6):
    """Các từ quanh khía cạnh (khía cạnh thay bằng ASP), để mô hình biết đang hỏi về khía cạnh nào."""
    t = text.lower(); a = asp.lower()
    i = t.find(a)
    if i < 0:
        return t
    trai, phai = t[:i].split()[-w:], t[i + len(a):].split()[:w]
    return " ".join(trai + ["ASP"] + phai)


def mau(rows, co_nhan):
    X, Y, khoa = [], [], []
    for r in rows:
        for a in r["aspects"]:
            ten = a["aspect"] if co_nhan else a["term"]
            X.append(cua_so(r["text"], ten)); khoa.append((r["id"], ten))
            if co_nhan:
                Y.append((a["valence"], a["arousal"]))
    return X, np.array(Y), khoa


def rmse_va(gold_rows, pred):
    """pred: {(id, aspect): (v, a)}. Giống metric của cuộc thi."""
    e = [(pred[(r["id"], a["aspect"])][0] - a["valence"]) ** 2 + (pred[(r["id"], a["aspect"])][1] - a["arousal"]) ** 2
         for r in gold_rows for a in r["aspects"]]
    return float(np.sqrt(np.mean(e)))


train, dev = doc(D / "train.jsonl"), doc(D / "dev.jsonl")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def fit_predict(tr_rows, X_te):
    X, Y, _ = mau(tr_rows, True)
    M = vec.fit_transform(X)
    P = np.column_stack([Ridge(alpha=1.0).fit(M, Y[:, k]).predict(vec.transform(X_te)) for k in range(2)])
    return np.clip(P, 1, 9).round(2)


# %% [markdown]
# ## Tự chấm trên dev

# %%
Xd, _, kd = mau(dev, True)
Pd = fit_predict(train, Xd)
print(f"dev RMSE_VA = {rmse_va(dev, dict(zip(kd, Pd))):.4f}")
v_tb, a_tb = np.mean([a["valence"] for r in train for a in r["aspects"]]), np.mean([a["arousal"] for r in train for a in r["aspects"]])
print(f"(đoán trung bình train: {rmse_va(dev, {k: (v_tb, a_tb) for k in kd}):.4f})")

# %% [markdown]
# ## Dự đoán test (public) và private_test (nếu đã giải nén), train lại trên train + dev

# %%
test = doc(D / "test.jsonl")
for thu in [Path("private_test.jsonl"), D / "private_test.jsonl"]:
    if thu.exists():
        test += doc(thu); break
Xt, _, kt = mau(test, False)
Pt = dict(zip(kt, fit_predict(train + dev, Xt)))
out = [{"id": r["id"], "aspects": [{"aspect": a["term"], "valence": float(Pt[(r["id"], a["term"])][0]),
                                    "arousal": float(Pt[(r["id"], a["term"])][1])} for a in r["aspects"]]} for r in test]
with open("pred_test.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
pd.DataFrame({"id": [o["id"] for o in out], "aspects": [json.dumps(o["aspects"], ensure_ascii=False) for o in out]}).to_csv("submission.csv", index=False)
print(f"ghi pred_test.jsonl và submission.csv: {len(out)} câu")
