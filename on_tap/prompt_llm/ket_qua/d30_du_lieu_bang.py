import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

def seed_all(seed):
    np.random.seed(seed)

seed_all(42)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

for df in [train, test]:
    df["tre_x_ho_tro"] = df.thanh_toan_tre * df.cuoc_goi_ho_tro

num = ["tuoi", "thu_nhap", "so_nam_kh", "cuoc_goi_ho_tro", "thanh_toan_tre", "diem_hai_long", "tre_x_ho_tro"]
cat = ["thanh_pho", "nghe_nghiep", "goi_cuoc"]

X = train[num + cat]
y = train.churn
Xt = test[num + cat]

model_a = make_pipeline(
    ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=20)), cat)
    ]),
    LogisticRegression(max_iter=2000, random_state=42)
)

model_b = make_pipeline(
    ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat),
        ("num", "passthrough", num)
    ]),
    HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, categorical_features=[0, 1, 2], random_state=42)
)

cv = StratifiedKFold(5, shuffle=True, random_state=42)

oof_a = cross_val_predict(model_a, X, y, cv=cv, method="predict_proba")[:, 1]
oof_b = cross_val_predict(model_b, X, y, cv=cv, method="predict_proba")[:, 1]

print(f"AUROC Model A: {roc_auc_score(y, oof_a)}")
print(f"AUROC Model B: {roc_auc_score(y, oof_b)}")

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

Z = np.column_stack([logit(oof_a), logit(oof_b)])

meta = LogisticRegression()

meta_cv = cross_val_predict(meta, Z, y, cv=cv, method="predict_proba")[:, 1]
print(f"Meta AUROC: {roc_auc_score(y, meta_cv)}")

meta.fit(Z, y)

model_a.fit(X, y)
model_b.fit(X, y)

prob_a = model_a.predict_proba(Xt)[:, 1]
prob_b = model_b.predict_proba(Xt)[:, 1]

Zt = np.column_stack([logit(prob_a), logit(prob_b)])

prob = meta.predict_proba(Zt)[:, 1]

submission = pd.DataFrame({"id": test.id, "prob": prob})
submission.to_csv("submission.csv", index=False)
