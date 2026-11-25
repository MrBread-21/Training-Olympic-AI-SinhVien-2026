# %% [markdown]
# # Baseline - Dự đoán vỡ nợ (Home Credit, dữ liệu bảng nhiều file)
#
# Chỉ dùng `application_train.csv` (chưa dùng bureau và previous_application): cột số giữ nguyên, cột chữ mã hoá thành số,
# HistGradientBoosting của sklearn. Tự chấm AUROC trên 3 tháng cuối của train (chia theo thời gian, giống cách chia test),
# rồi train lại trên toàn bộ train và ghi `submission.csv` (`id,score`).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("application_train.csv")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)

tr = pd.read_csv(D / "application_train.csv")
te = pd.read_csv(D / "application_test.csv")
for thu in [Path("application_private_test.csv"), D / "application_private_test.csv"]:
    if thu.exists():
        te = pd.concat([te, pd.read_csv(thu)], ignore_index=True); break
print("train", tr.shape, "test", te.shape, "tỉ lệ vỡ nợ", round(tr.label.mean(), 4))

cot_chu = [c for c in tr.columns if tr[c].dtype == object and c not in ("id", "application_date")]
for c in cot_chu:
    ma = {v: i for i, v in enumerate(tr[c].dropna().unique())}
    tr[c], te[c] = tr[c].map(ma), te[c].map(ma)
X = [c for c in tr.columns if c not in ("id", "label", "application_date")]


def mo_hinh():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                          categorical_features=[X.index(c) for c in cot_chu], random_state=42)


# %% [markdown]
# ## Tự chấm: train trên các tháng đầu, kiểm tra trên 3 tháng cuối của train

# %%
moc = tr.application_date >= "2020-04"
m = mo_hinh().fit(tr.loc[~moc, X], tr.loc[~moc, "label"])
print(f"val AUROC (04-06/2020) = {roc_auc_score(tr.loc[moc, 'label'], m.predict_proba(tr.loc[moc, X])[:, 1]):.4f}")

# %% [markdown]
# ## Train lại trên toàn bộ train, dự đoán test

# %%
m = mo_hinh().fit(tr[X], tr.label)
pd.DataFrame({"id": te.id, "score": m.predict_proba(te[X])[:, 1]}).to_csv("submission.csv", index=False)
print(f"ghi submission.csv: {len(te)} hồ sơ")
