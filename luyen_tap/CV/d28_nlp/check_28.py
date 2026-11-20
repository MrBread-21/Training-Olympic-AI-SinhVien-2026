"""Chấm bài 28.  python check_28.py [bai_a.py] [bai_a2.py] [bai_b.py] [--khong-chay]
A: TF-IDF + logistic (submission_a.csv). A2: TextCNN/BiLSTM tự train (submission_a2.csv). B: BM25 (submission_b.csv)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

NG_A, NG_A2, NG_B = 0.87, 0.85, 0.85   # lời giải mẫu 0.91 / 0.90 / 0.93; quên NFC chỉ ~0.80
args = [a for a in sys.argv[1:] if not a.startswith("--")] + ["bai_a.py", "bai_a2.py", "bai_b.py"][len([a for a in sys.argv[1:] if not a.startswith("--")]):]
fa, fa2, fb = [Path(a).resolve() for a in args[:3]]
can_chay = "--khong-chay" not in sys.argv
B = Bang("Bài 28 - NLP")
ans_a = doc_dap_an(HERE, "a_test.csv")

print("A - TF-IDF + logistic")
kiem_tra_code(B, fa, torch=False, nfc=True, tu_khoa=[(r"char", "có n-gram ký tự (char / char_wb)")])
if can_chay: chay(B, fa, timeout_phut=10)
d = doc_csv(B, fa.parent / "submission_a.csv", ["id", "label"], ans_a.id)
f = f1_score(ans_a.label, d.label, average="macro")
B.ok(f >= NG_A, f"A: macro-F1 = {f:.4f} (ngưỡng {NG_A})")

print("\nA2 - mạng nơ-ron tự train")
kiem_tra_code(B, fa2, nfc=True, tu_khoa=[(r"Embedding", "có nn.Embedding (vocab tự xây)"),
                                         (r"argsort|sorted\(", "có bucketing (sắp theo độ dài)")])
if can_chay: chay(B, fa2, timeout_phut=10)
d = doc_csv(B, fa2.parent / "submission_a2.csv", ["id", "label"], ans_a.id)
f = f1_score(ans_a.label, d.label, average="macro")
B.ok(f >= NG_A2, f"A2: macro-F1 = {f:.4f} (ngưỡng {NG_A2})")

print("\nB - BM25")
kiem_tra_code(B, fb, torch=False, nfc=True, tu_khoa=[(r"idf", "có IDF"), (r"k1", "có tham số k1, b của BM25")])
if can_chay: chay(B, fb, timeout_phut=5)
ans_b = doc_dap_an(HERE, "b_test.csv")
d = doc_csv(B, fb.parent / "submission_b.csv", ["id", "passage_ids"], ans_b.id)
lists = [str(s).split() for s in d.passage_ids]
r1 = np.mean([l[:1] == [g] for l, g in zip(lists, ans_b.passage_id)]); r5 = np.mean([g in l[:5] for l, g in zip(lists, ans_b.passage_id)])
B.ok(r1 >= NG_B, f"B: Recall@1 = {r1:.3f} (ngưỡng {NG_B}), Recall@5 = {r5:.3f}")
B.ket_thuc()
