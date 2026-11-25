"""Chấm bài 22.  python check_22.py [bai_lam.py] [--khong-chay]"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, doi_so, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

NGUONG = 0.90
B = Bang("Bài 22 - augment + nhãn nhiễu")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, tu_khoa=[(r"RandAugment|TrivialAugment|AutoAugment", "có RandAugment (hoặc tương đương)"),
                               (r"MixUp|CutMix|mixup|cutmix", "có MixUp/CutMix"),
                               (r"label_smoothing", "có label smoothing"),
                               (r"rot90|flip", "có TTA xoay/lật"),
                               (r"argsort|small.?loss|keep", "có chọn mẫu loss nhỏ (small-loss)")])
if can_chay:
    chay(B, bai, timeout_phut=25)
ans = doc_dap_an(HERE, "test.csv")
df = doc_csv(B, bai.parent / "submission.csv", ["id", "label"], ans.id)
B.ok(df.label.isin(set(ans.label)).all(), "nhãn thuộc 6 lớp hợp lệ")
f1 = f1_score(ans.label, df.label, average="macro")
B.ok(f1 >= NGUONG, f"macro-F1 test = {f1:.4f} (ngưỡng {NGUONG}; khung bài 21 để nguyên chỉ được khoảng 0.82)")
B.ket_thuc()
