# Bài 30 (30/11) - Dữ liệu bảng: ColumnTransformer, HistGradientBoosting, stacking OOF

**Thời gian mục tiêu: ≤ 35 phút.**

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **scikit-learn**: [ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#column-transformer-for-heterogeneous-data), [Imputation](https://scikit-learn.org/stable/modules/impute.html) (`add_indicator`), [Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) (`handle_unknown`, `min_frequency`), [Histogram-based Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) (giá trị thiếu, `categorical_features`), [Stacked generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization).
- **Ví dụ**: [Categorical Feature Support in Gradient Boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html) (so sánh one-hot, ordinal, native).
- **Khoá ngắn**: Kaggle Learn [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) (missing values, categorical variables, pipelines, cross-validation, data leakage).
- **Bài báo**: Wolpert, *Stacked Generalization*, Neural Networks 5(2), 1992 (ý tưởng: mô hình meta học trên dự đoán ngoài fold).
- **Trong máy**: `../../ON_TAP.md` phần 1.6 (dữ liệu bảng); `../../MSAD/main.py` docstring (dò trọng số ensemble kiểu Caruana, stacking chỉ giữ thành phần có ích).

## Bài 1 - Dự đoán khách hàng rời bỏ

- **Đầu vào**: `data/train.csv` (6000 dòng): `tuoi, thu_nhap, thanh_pho, nghe_nghiep, so_nam_kh, goi_cuoc, cuoc_goi_ho_tro, thanh_toan_tre, diem_hai_long`, nhãn `churn`; `data/test.csv` (3000 dòng). Có **ô trống** ở cột số và cột phân loại, **nhóm hiếm** ở `thanh_pho`, nhãn phụ thuộc **phi tuyến** và có **tương tác** giữa các cột.
- **Đầu ra**: `submission.csv` (`id,prob`), xác suất trong [0, 1].
- **Metric**: AUROC, **ngưỡng 0.73** (logistic một mình chỉ khoảng 0.70).
- **Khung bắt buộc**:
  1. Pipeline tuyến tính: `ColumnTransformer` (số: `SimpleImputer(median, add_indicator=True)` + `StandardScaler`; phân loại: impute + `OneHotEncoder(handle_unknown="ignore", min_frequency=...)`) + LogisticRegression.
  2. Pipeline cây: `OrdinalEncoder` (giá trị lạ, thiếu thành -1) + `HistGradientBoostingClassifier(categorical_features=...)`, để NaN cột số cho HGB tự xử lý.
  3. **OOF** 5-fold `StratifiedKFold` (`cross_val_predict(..., method="predict_proba")`) cho từng mô hình, in AUROC OOF.
  4. **Stacking**: LogisticRegression trên logit của các xác suất OOF; đánh giá stacking cũng bằng CV.
  5. Bản cuối: fit các mô hình gốc trên toàn bộ train, dự đoán test, đưa qua mô hình meta.
- Thử thêm một đặc trưng tương tác tự nghĩ ra và xem OOF thay đổi thế nào.

## Bài 2 - Viết lại nhanh bài 29 (≤ 35 phút)

Trong `d29_seq2seq`: `viet_lai.py`, chấm `python check_29.py viet_lai.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_30.py bai_lam.py
```

Lời giải mẫu: OOF logistic 0.70, HGB 0.74, stacking 0.745; test 0.75.
