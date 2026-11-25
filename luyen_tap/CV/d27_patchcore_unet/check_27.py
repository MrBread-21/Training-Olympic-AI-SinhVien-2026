"""Chấm bài 27.  python check_27.py [bai_a.py] [bai_b.py] [--khong-chay]
Phần A: submission_a.csv (id,score), AUROC. Phần B: masks_pred/<id>.png (0/255), Dice trung bình."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402
import cv2  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

NGUONG_AUC, NGUONG_DICE = 0.78, 0.80   # lời giải mẫu: AUROC ~0.84, Dice ~0.93
args = [a for a in sys.argv[1:] if not a.startswith("--")]
bai_a = Path(args[0] if args else "bai_a.py").resolve(); bai_b = Path(args[1] if len(args) > 1 else "bai_b.py").resolve()
can_chay = "--khong-chay" not in sys.argv
B = Bang("Bài 27 - PatchCore + U-Net")

print("Phần A - PatchCore")
kiem_tra_code(B, bai_a, tu_khoa=[(r"layer2", "dùng đặc trưng tầng giữa (layer2/layer3)"),
                                 (r"cdist|topk|kneighbors|min\(", "chấm điểm bằng khoảng cách kNN tới memory bank")])
if can_chay:
    chay(B, bai_a, timeout_phut=10)
ans = doc_dap_an(HERE, "a_test.csv")
df = doc_csv(B, bai_a.parent / "submission_a.csv", ["id", "score"], ans.id)
auc = roc_auc_score(ans.anomaly, df.score.astype(float))
B.ok(auc >= NGUONG_AUC, f"AUROC phần A = {auc:.4f} (ngưỡng {NGUONG_AUC})")

print("\nPhần B - U-Net")
kiem_tra_code(B, bai_b, tu_khoa=[(r"cat\(", "decoder nối skip connection (torch.cat)"),
                                 (r"dice|Dice", "có Dice (loss hoặc đo trên val)")])
if can_chay:
    chay(B, bai_b, timeout_phut=15)
ds = []
for p in sorted((HERE / ".dap_an/b_masks").glob("*.png")):
    q = bai_b.parent / "masks_pred" / p.name
    pr = cv2.imread(str(q), 0) if q.exists() else None
    if pr is None or pr.shape != (64, 64):
        B.ok(False, f"thiếu hoặc sai kích thước masks_pred/{p.name}"); B.ket_thuc()
    t, pr = cv2.imread(str(p), 0) > 127, pr > 127
    ds.append(1.0 if t.sum() + pr.sum() == 0 else 2 * (t & pr).sum() / (t.sum() + pr.sum()))
B.ok(True, f"đủ {len(ds)} mặt nạ 64x64")
B.ok(np.mean(ds) >= NGUONG_DICE, f"Dice trung bình phần B = {np.mean(ds):.4f} (ngưỡng {NGUONG_DICE})")
B.ket_thuc()
