"""Chấm bài 26.  python check_26.py [bai_lam.py] [--khong-chay]
Điểm mỗi video: binh_thuong được 1 nếu đoán binh_thuong; sự kiện được tIoU khung nếu đúng nhãn. Trung bình theo lớp rồi trung bình đều."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, doi_so, kiem_tra_code, chay, doc_dap_an  # noqa: E402

NGUONG = 0.50
B = Bang("Bài 26 - video: phân loại + định vị đoạn")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, tu_khoa=[(r"linspace", "lấy N khung rải đều (np.linspace)"),
                               (r"np\.save|savez", "lưu cache đặc trưng"),
                               (r"Logistic", "phân loại bằng logistic trên đặc trưng gộp thời gian")])
if can_chay:
    chay(B, bai, timeout_phut=15)
ans = pd.read_csv(HERE / ".dap_an/test.csv"); nf = pd.read_csv(HERE / "data/video_info.csv").set_index("video_id").frames
p = bai.parent / "submission.csv"
if not B.ok(p.exists(), "có submission.csv"):
    B.ket_thuc()
df = pd.read_csv(p)
B.ok(list(df.columns) == ["video_id", "label", "start_frame", "end_frame"], "tiêu đề video_id,label,start_frame,end_frame")
B.ok(df.video_id.is_unique and set(df.video_id) == set(ans.video_id), "đủ video test, mỗi video một dòng")
df = df.set_index("video_id").loc[ans.video_id]
B.ok(df.label.isin(set(ans.label)).all(), "nhãn hợp lệ")
ab = df.label != "binh_thuong"
hop_le = ab.sum() == 0 or (df[ab].start_frame.notna().all() and (df[ab].start_frame >= 1).all()
                          and (df[ab].start_frame <= df[ab].end_frame).all() and (df[ab].end_frame.values <= nf.loc[df[ab].index].values).all())
B.ok(hop_le, "video có sự kiện: 1 <= start_frame <= end_frame <= số khung")
diem = []
for (_, a), (_, d) in zip(ans.iterrows(), df.iterrows()):
    if a.label == "binh_thuong": diem.append(float(d.label == "binh_thuong"))
    elif d.label != a.label: diem.append(0.0)
    else:
        inter = min(a.end_frame, d.end_frame) - max(a.start_frame, d.start_frame) + 1
        union = (a.end_frame - a.start_frame + 1) + (d.end_frame - d.start_frame + 1) - max(inter, 0)
        diem.append(max(inter, 0) / union)
ans["diem"] = diem
theo_lop = ans.groupby("label").diem.mean()
B.ok(theo_lop.mean() >= NGUONG, f"điểm macro = {theo_lop.mean():.4f} (ngưỡng {NGUONG}) | theo lớp {theo_lop.round(3).to_dict()}")
B.ket_thuc()
