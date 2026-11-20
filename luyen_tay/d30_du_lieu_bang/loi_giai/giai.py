import random
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
seed_all(42)
tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y, X, Xte = tr.churn.values, tr.drop(columns=["id", "churn"]), te.drop(columns=["id"])
CAT = [c for c in X.columns if X[c].dtype == object]; NUM = [c for c in X.columns if c not in CAT]
X["goi_x_nam"] = X.goi_cuoc.fillna("?") + "_" + (X.so_nam_kh < 3).astype(str)          # đặc trưng tương tác
Xte["goi_x_nam"] = Xte.goi_cuoc.fillna("?") + "_" + (Xte.so_nam_kh < 3).astype(str); CAT.append("goi_x_nam")

lr = make_pipeline(ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=30)), CAT)]),
    LogisticRegression(C=0.5, max_iter=3000))
hgb = make_pipeline(ColumnTransformer([
        ("num", "passthrough", NUM),                                                   # HGB tự xử lý NaN
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CAT)]),
    HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0,
                                   categorical_features=list(range(len(NUM), len(NUM) + len(CAT))), random_state=42))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = {}
for name, m in [("logistic", lr), ("HGB", hgb)]:
    oof[name] = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]       # dự đoán ngoài fold
    print(f"OOF AUROC {name}: {roc_auc_score(y, oof[name]):.4f}")

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
S = np.c_[logit(oof["logistic"]), logit(oof["HGB"])]
meta = LogisticRegression()
print(f"OOF AUROC stacking: {roc_auc_score(y, cross_val_predict(meta, S, y, cv=cv, method='predict_proba')[:, 1]):.4f}")
meta.fit(S, y); print("trọng số meta:", meta.coef_.round(3))

lr.fit(X, y); hgb.fit(X, y)                                                            # bản cuối: train trên toàn bộ dữ liệu
Ste = np.c_[logit(lr.predict_proba(Xte)[:, 1]), logit(hgb.predict_proba(Xte)[:, 1])]
pd.DataFrame({"id": te.id, "prob": meta.predict_proba(Ste)[:, 1]}).to_csv("submission.csv", index=False)
print("đã ghi submission.csv")
