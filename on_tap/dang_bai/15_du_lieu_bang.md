# Dạng 15 - Dữ liệu bảng (tabular)

## 1. Nhận dạng đề

- Một hoặc nhiều file CSV; mỗi dòng của bảng chính là một đối tượng (khách hàng, giao dịch, căn nhà...), dự đoán một cột (phân loại hoặc hồi quy). Bảng phụ nối bằng khoá (một đối tượng nhiều dòng).
- Ví dụ: HomeCredit_Tabular (tự host: 3 bảng, chia theo thời gian, AUROC), Kaggle Playground, dự đoán giá nhà, dự đoán khách rời bỏ.

## 2. Metric

AUROC / logloss / F1 (phân loại), RMSE / RMSLE / MAE (hồi quy). RMSLE: train trên `log1p(y)` rồi `expm1`.

## 3. Lộ trình

| Bước | Việc | Thời gian |
|---|---|---|
| 0 | Baseline BTC, nộp | 10 phút |
| 1 | EDA: kiểu cột, tỉ lệ thiếu, phân bố nhãn, cột id / thời gian, khác biệt train và test (adversarial validation: phân loại train vs test, AUROC cao là có lệch) | 15 phút |
| 2 | **LightGBM / XGBoost / CatBoost** mặc định trên bảng chính (cột chữ để dạng category), CV 5-fold (theo thời gian nếu có), early stopping | 15 phút |
| 3 | **Feature engineering**: tỉ lệ, hiệu, tương tác; gộp bảng phụ theo khoá (count, mean, max, min, sum, tỉ lệ theo trạng thái); target encoding **trong fold** | 1-2 giờ, phần lớn điểm |
| 4 | Tinh chỉnh vừa phải (num_leaves, learning_rate nhỏ + nhiều cây, feature_fraction), blend 2-3 thư viện | |

## 4. Khung code

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
cat_cols = [c for c in X.columns if X[c].dtype == object]
for c in cat_cols:
    X[c] = X[c].astype("category"); Xte[c] = pd.Categorical(Xte[c], categories=X[c].cat.categories)

oof, pred = np.zeros(len(X)), np.zeros(len(Xte))
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=1.0, random_state=f, verbose=-1)
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]; pred += m.predict_proba(Xte)[:, 1] / 5
print("OOF AUROC", roc_auc_score(y, oof))
```

Không có LightGBM: `sklearn.ensemble.HistGradientBoostingClassifier(categorical_features=...)` gần tương đương.

Gộp bảng phụ:

```python
agg = phu.groupby("id").agg(so_dong=("id", "size"), tien_tb=("AMT_CREDIT_SUM", "mean"), tien_max=("AMT_CREDIT_SUM", "max"),
                            qua_han_max=("CREDIT_DAY_OVERDUE", "max"), ngay_gan_nhat=("DAYS_CREDIT", "max"))
agg["ti_le_dang_no"] = phu.assign(a=phu.CREDIT_ACTIVE.eq("Active")).groupby("id").a.mean()
X = bang_chinh.merge(agg, on="id", how="left")
```

Target encoding trong fold (tránh rò rỉ):

```python
def te_fold(tr_df, va_df, col, y_col, a=20):
    g = tr_df.groupby(col)[y_col].agg(["mean", "count"]); prior = tr_df[y_col].mean()
    s = (g["mean"] * g["count"] + prior * a) / (g["count"] + a)          # làm mượt về trung bình
    return va_df[col].map(s).fillna(prior)
```

## 5. Nút vặn

1. Đặc trưng nghiệp vụ (tỉ lệ khoản vay / thu nhập, tuổi, thời gian làm việc) và gộp bảng phụ.
2. Validation đúng (theo thời gian / theo nhóm); tin OOF hơn LB khi test public nhỏ.
3. Giá trị đặc biệt (`365243` nghĩa là thiếu trong Home Credit), cột ngày thành tuổi / khoảng cách.
4. Learning rate nhỏ + nhiều cây + early stopping.
5. Blend LightGBM + CatBoost + XGBoost (trung bình hạng với AUROC).
6. Lệch lớp: `scale_pos_weight` chỉ cần khi metric là F1 / logloss; AUROC không đổi theo trọng số lớp nhiều.

## 6. Bẫy

- Rò rỉ: đặc trưng dùng thông tin tương lai, target encoding trên toàn train, chuẩn hoá theo cả test.
- Cột id mang thông tin thứ tự (tăng theo thời gian): bỏ, hoặc dùng có ý thức.
- Category trong test không có trong train: `pd.Categorical(..., categories=...)` để thành NaN thay vì lỗi.
- Gộp bảng phụ bằng `merge` mà quên `how="left"`: mất dòng của bảng chính.

## 7. Bài luyện

- Code tay: [d30_du_lieu_bang](../../luyen_tap/CV/d30_du_lieu_bang/) (ColumnTransformer, encoding, HistGradientBoosting, OOF, stacking).
- Thi thử: [HomeCredit_Tabular](../../selfhost_kaggle/HomeCredit_Tabular/) (baseline chỉ bảng chính: 0,761 / 0,747; gộp bảng phụ là phần đáng làm nhất).

## 8. Đọc thêm

- Ke và cộng sự, *LightGBM* (NeurIPS 2017); Prokhorenkova và cộng sự, *CatBoost* ([arXiv:1706.09516](https://arxiv.org/abs/1706.09516)).
- Grinsztajn và cộng sự, *Why do tree-based models still outperform deep learning on tabular data?* ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)).
- Lời giải hạng cao cuộc thi Home Credit Default Risk trên Kaggle (thảo luận "1st place solution"): gộp bảng phụ theo nhiều cửa sổ thời gian.
