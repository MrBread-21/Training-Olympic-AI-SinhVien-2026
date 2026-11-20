"""Chấm bài 23.  python check_23.py [bai_lam.py] [--khong-chay]"""
import sys, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, doi_so, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

NGUONG = 0.85
B = Bang("Bài 23 - đặc trưng đóng băng + CV theo nhóm")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, tu_khoa=[(r"GroupKFold", "chia fold theo nhóm (GroupKFold / StratifiedGroupKFold)"),
                               (r"confusion_matrix", "in ma trận nhầm lẫn"),
                               (r"np\.save|savez|to_pickle|torch\.save", "lưu cache đặc trưng")])
t0 = time.time()
if can_chay:
    chay(B, bai, timeout_phut=15)
caches = [p for p in bai.parent.iterdir() if p.suffix in (".npy", ".npz", ".pt", ".pkl")]
B.ok(caches, f"có file cache đặc trưng trong thư mục ({', '.join(p.name for p in caches) or 'không thấy'})")
tr = pd.read_csv(HERE / "data/train.csv")
oof = doc_csv(B, bai.parent / "oof.csv", ["id", "label"], tr.id)
f_oof = f1_score(tr.label, oof.label, average="macro")
ans = doc_dap_an(HERE, "test.csv")
df = doc_csv(B, bai.parent / "submission.csv", ["id", "label"], ans.id)
f_te = f1_score(ans.label, df.label, average="macro")
B.ok(f_te >= NGUONG, f"macro-F1 test = {f_te:.4f} (ngưỡng {NGUONG})")
B.ok(f_oof - f_te <= 0.04, f"OOF macro-F1 = {f_oof:.4f}, lệch test {f_oof - f_te:+.4f}: OOF cao hơn test quá 0.04 "
                           "thường là do chia fold không theo nhóm (rò rỉ)", canh_bao=True)
B.ket_thuc()
