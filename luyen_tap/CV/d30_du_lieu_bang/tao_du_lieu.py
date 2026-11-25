"""Tạo dữ liệu bài 30: dự đoán khách hàng rời bỏ (churn). Cột số có ô trống, cột phân loại có nhóm hiếm,
nhãn phụ thuộc phi tuyến và có tương tác. 6000 dòng train, 3000 dòng test. CPU dưới 1 giây."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(3001)
D, A = HERE / "data", HERE / ".dap_an"
D.mkdir(exist_ok=True); A.mkdir(exist_ok=True)
N = 9000
TP = np.array(["Hà Nội", "TP.HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Huế", "Nha Trang", "Vinh", "Đà Lạt", "Lào Cai", "Cà Mau", "Kon Tum"])
p_tp = np.array([30, 30, 8, 8, 6, 5, 4, 3, 2.5, 1.5, 1, 1]); p_tp /= p_tp.sum()
NGHE = np.array(["văn phòng", "công nhân", "sinh viên", "kinh doanh", "nông dân", "giáo viên", "y tế", "khác"])
df = pd.DataFrame({
    "tuoi": rng.integers(18, 80, N).astype(float),
    "thu_nhap": np.round(np.exp(rng.normal(2.3, 0.6, N)), 2),                     # triệu đồng / tháng
    "thanh_pho": rng.choice(TP, N, p=p_tp),
    "nghe_nghiep": rng.choice(NGHE, N),
    "so_nam_kh": rng.integers(0, 21, N),
    "goi_cuoc": rng.choice(["co_ban", "tieu_chuan", "cao_cap"], N, p=[0.5, 0.35, 0.15]),
    "cuoc_goi_ho_tro": rng.poisson(1.5, N),
    "thanh_toan_tre": rng.integers(0, 2, N),
    "diem_hai_long": rng.integers(1, 11, N).astype(float),
})
z = (-1.6 + 0.9 * (df.cuoc_goi_ho_tro >= 4) + 1.3 * ((df.goi_cuoc == "co_ban") & (df.so_nam_kh < 3))
     + 0.03 * (df.tuoi - 45) ** 2 / 10 - 0.25 * (df.diem_hai_long - 5.5) + 0.8 * df.thanh_toan_tre * (df.thu_nhap < 7)
     + df.thanh_pho.map({"Kon Tum": 1.2, "Cà Mau": 1.0, "Lào Cai": 0.8, "TP.HCM": 0.2}).fillna(0) + 0.6 * (df.nghe_nghiep == "sinh viên")
     - 0.05 * df.so_nam_kh + rng.normal(0, 0.8, N))
df["churn"] = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)
for c, p in [("tuoi", 0.05), ("thu_nhap", 0.08), ("diem_hai_long", 0.10), ("nghe_nghiep", 0.04)]:
    df.loc[rng.random(N) < p, c] = np.nan
df.insert(0, "id", [f"kh{i:05d}" for i in range(N)])
df[:6000].to_csv(D / "train.csv", index=False)
df[6000:].drop(columns="churn").to_csv(D / "test.csv", index=False)
df[6000:][["id", "churn"]].to_csv(A / "test.csv", index=False)
df[6000:][["id"]].assign(prob=0.5).to_csv(D / "sample_submission.csv", index=False)
print("xong: 6000 train / 3000 test, tỉ lệ churn %.3f" % df.churn.mean())
