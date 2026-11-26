"""Lời giải HomeCredit_Tabular: gộp bảng phụ (bureau, previous_application) theo id + đặc trưng tỉ lệ + HistGradientBoosting,
validation theo thời gian (3 tháng cuối của train). Chỉ dùng sklearn (máy thi có thể không có LightGBM).
Chạy: python solve.py <thư_mục_dữ_liệu> (và application_private_test.csv nếu đã giải nén)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
t0 = time.time()
tr = pd.read_csv(D / "application_train.csv"); te = pd.read_csv(D / "application_test.csv")
for p in [Path("application_private_test.csv"), D / "application_private_test.csv"]:
    if p.exists(): te = pd.concat([te, pd.read_csv(p)], ignore_index=True); break
bu = pd.read_csv(D / "bureau.csv"); pr = pd.read_csv(D / "previous_application.csv")
print(f"đọc xong {time.time() - t0:.0f}s: train {tr.shape}, test {te.shape}, bureau {bu.shape}, previous {pr.shape}")

# ---------- bảng phụ
bu["dang_no"] = (bu.CREDIT_ACTIVE == "Active").astype(float)
bu["qua_han"] = (bu.CREDIT_DAY_OVERDUE > 0).astype(float)
bu["ti_le_no"] = bu.AMT_CREDIT_SUM_DEBT / bu.AMT_CREDIT_SUM.replace(0, np.nan)
g = bu.groupby("id")
f_bu = pd.DataFrame({
    "bu_so": g.size(), "bu_dang_no": g.dang_no.mean(), "bu_so_dang_no": g.dang_no.sum(), "bu_qua_han": g.qua_han.mean(),
    "bu_qua_han_max": g.CREDIT_DAY_OVERDUE.max(), "bu_ngay_moi_nhat": g.DAYS_CREDIT.max(), "bu_ngay_tb": g.DAYS_CREDIT.mean(),
    "bu_tien_tong": g.AMT_CREDIT_SUM.sum(), "bu_no_tong": g.AMT_CREDIT_SUM_DEBT.sum(), "bu_ti_le_no_tb": g.ti_le_no.mean(),
    "bu_qua_han_tien": g.AMT_CREDIT_SUM_OVERDUE.sum(), "bu_gia_han": g.CNT_CREDIT_PROLONG.sum(),
    "bu_ket_thuc_tb": g.DAYS_CREDIT_ENDDATE.mean()})
for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
    pr[c] = pr[c].replace(365243, np.nan)                       # 365243 nghĩa là không có
pr["duyet"] = (pr.NAME_CONTRACT_STATUS == "Approved").astype(float)
pr["tu_choi"] = (pr.NAME_CONTRACT_STATUS == "Refused").astype(float)
pr["ti_le_cap"] = pr.AMT_CREDIT / pr.AMT_APPLICATION.replace(0, np.nan)
g = pr.groupby("id")
f_pr = pd.DataFrame({
    "pr_so": g.size(), "pr_duyet": g.duyet.mean(), "pr_tu_choi": g.tu_choi.mean(), "pr_so_tu_choi": g.tu_choi.sum(),
    "pr_tien_tb": g.AMT_CREDIT.mean(), "pr_tien_max": g.AMT_CREDIT.max(), "pr_ti_le_cap_tb": g.ti_le_cap.mean(),
    "pr_ti_le_cap_min": g.ti_le_cap.min(), "pr_ngay_moi_nhat": g.DAYS_DECISION.max(), "pr_tra_gop_tb": g.AMT_ANNUITY.mean(),
    "pr_ky_han_tb": g.CNT_PAYMENT.mean(), "pr_lai_tb": g.RATE_DOWN_PAYMENT.mean()})
pr_moi = pr.sort_values("DAYS_DECISION").groupby("id").tail(1).set_index("id")   # hồ sơ trước gần nhất
f_pr["pr_gan_tu_choi"] = pr_moi.tu_choi; f_pr["pr_gan_tien"] = pr_moi.AMT_CREDIT


def dac_trung(a):
    a = a.merge(f_bu, left_on="id", right_index=True, how="left").merge(f_pr, left_on="id", right_index=True, how="left")
    a["vay_thu_nhap"] = a.AMT_CREDIT / a.AMT_INCOME_TOTAL; a["tra_thu_nhap"] = a.AMT_ANNUITY / a.AMT_INCOME_TOTAL
    a["vay_hang"] = a.AMT_CREDIT / a.AMT_GOODS_PRICE; a["ky_han"] = a.AMT_CREDIT / a.AMT_ANNUITY
    a["DAYS_EMPLOYED"] = a.DAYS_EMPLOYED.replace(365243, np.nan); a["lam_tuoi"] = a.DAYS_EMPLOYED / a.DAYS_BIRTH
    ext = a[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    a["ext_tb"], a["ext_min"], a["ext_tich"] = ext.mean(1), ext.min(1), ext.prod(1, min_count=1)
    a["no_thu_nhap"] = a.bu_no_tong / a.AMT_INCOME_TOTAL
    return a


tr, te = dac_trung(tr), dac_trung(te)
cot_chu = [c for c in tr.columns if tr[c].dtype == object and c not in ("id", "application_date")]
for c in cot_chu:
    ma = {v: i for i, v in enumerate(tr[c].dropna().unique())}; tr[c], te[c] = tr[c].map(ma), te[c].map(ma)
X = [c for c in tr.columns if c not in ("id", "label", "application_date")]
mo_hinh = lambda: HistGradientBoostingClassifier(max_iter=1200, learning_rate=0.03, max_leaf_nodes=48, min_samples_leaf=80,
                                                  l2_regularization=2.0, categorical_features=[X.index(c) for c in cot_chu],
                                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=60, random_state=42)
moc = tr.application_date >= "2020-04"
m = mo_hinh().fit(tr.loc[~moc, X], tr.loc[~moc, "label"])
print(f"val AUROC (04-06/2020) = {roc_auc_score(tr.loc[moc, 'label'], m.predict_proba(tr.loc[moc, X])[:, 1]):.4f}, {m.n_iter_} cây, {time.time() - t0:.0f}s")
diem = np.zeros(len(te))
for s in range(3):                                               # bản cuối: toàn bộ train, trung bình 3 seed
    m = mo_hinh().set_params(random_state=s).fit(tr[X], tr.label); diem += m.predict_proba(te[X])[:, 1] / 3
pd.DataFrame({"id": te.id, "score": diem}).to_csv("submission.csv", index=False)
print(f"đã ghi submission.csv: {len(te)} hồ sơ, {time.time() - t0:.0f}s")
