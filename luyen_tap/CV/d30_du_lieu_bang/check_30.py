"""Chấm bài 30.  python check_30.py [bai_lam.py] [--khong-chay]"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, doi_so, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

NGUONG = 0.73   # lời giải mẫu 0.75; logistic một mình chỉ ~0.70
B = Bang("Bài 30 - dữ liệu bảng")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, torch=False, tu_khoa=[(r"ColumnTransformer", "dùng ColumnTransformer"),
                                            (r"OneHotEncoder|OrdinalEncoder|TargetEncoder", "mã hoá cột phân loại"),
                                            (r"HistGradientBoosting", "có HistGradientBoosting"),
                                            (r"cross_val_predict|oof", "có dự đoán OOF"),
                                            (r"meta|stack", "có stacking trên OOF")])
if can_chay:
    chay(B, bai, timeout_phut=10)
ans = doc_dap_an(HERE, "test.csv")
df = doc_csv(B, bai.parent / "submission.csv", ["id", "prob"], ans.id)
B.ok(df.prob.between(0, 1).all(), "prob nằm trong [0, 1]")
auc = roc_auc_score(ans.churn.astype(int), df.prob)
B.ok(auc >= NGUONG, f"AUROC test = {auc:.4f} (ngưỡng {NGUONG})")
B.ket_thuc()
