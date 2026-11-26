"""Chấm bài 01.  python check_01.py [bai_lam.py] [--khong-chay]
Nhận dạng từ khoá trong âm thanh (submission.csv: id,label). Metric: macro-F1 trên 9 lớp."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, kiem_tra_code, chay, doc_csv, doc_dap_an, doi_so  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

NG = 0.85   # lời giải mẫu khoảng 0.92; cùng lời giải nhưng không trộn tạp âm khi train: 0.35
path, can_chay = doi_so()
B = Bang("Bài 01 - âm thanh")
kiem_tra_code(B, path, tu_khoa=[(r"stft", "tự tính spectrogram bằng torch.stft"), (r"snr|SNR", "trộn tạp âm theo SNR"),
                                (r"speaker", "val chia theo người nói")])
if can_chay: chay(B, path, timeout_phut=10)
ans = doc_dap_an(HERE, "test.csv")
d = doc_csv(B, path.parent / "submission.csv", ["id", "label"], ans.id)
f = f1_score(ans.label, d.label, average="macro")
B.ok(f >= NG, f"macro-F1 = {f:.4f} (ngưỡng {NG})")
B.ket_thuc()
