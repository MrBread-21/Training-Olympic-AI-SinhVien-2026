"""Chấm bài 21.  python check_21.py [bai_lam.py] [--khong-chay]"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, doi_so, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

NGUONG = 0.95
B = Bang("Bài 21 - phân loại ảnh")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, tu_khoa=[(r"torch\.save", "lưu checkpoint tốt nhất (torch.save)"),
                               (r"load_state_dict", "nạp lại checkpoint tốt nhất trước khi dự đoán"),
                               (r"DataLoader", "dùng DataLoader")])
if can_chay:
    chay(B, bai, timeout_phut=20)
ans = doc_dap_an(HERE, "test.csv")
df = doc_csv(B, bai.parent / "submission.csv", ["id", "label"], ans.id)
B.ok(df.label.isin(set(ans.label)).all(), "nhãn thuộc 6 lớp hợp lệ")
f1 = f1_score(ans.label, df.label, average="macro")
B.ok(f1 >= NGUONG, f"macro-F1 test = {f1:.4f} (ngưỡng {NGUONG})")
B.ket_thuc()
