"""Chấm bài 24.  python check_24.py [bai_lam.py] [--khong-chay]"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, doi_so, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402

NGUONG_R1 = 0.75
B = Bang("Bài 24 - truy xuất ảnh")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, tu_khoa=[(r"GeM|pow\(\s*1\s*/", "có GeM pooling"),
                               (r"arcface|ArcFace|acos", "có ArcFace (cos(theta + m))"),
                               (r"normalize", "chuẩn hoá L2 embedding trước khi tính cosine"),
                               (r"recall|Recall|R@1", "tự đo Recall@K trên val")])
if can_chay:
    chay(B, bai, timeout_phut=20)
g, q = doc_dap_an(HERE, "gallery.csv"), doc_dap_an(HERE, "query.csv")
df = doc_csv(B, bai.parent / "submission.csv", ["id", "matches"], q.id)
lists = [str(s).split() for s in df.matches]
gset = set(g.id)
B.ok(all(len(l) >= 5 and len(set(l)) == len(l) for l in lists), "mỗi query có ít nhất 5 kết quả, không trùng")
B.ok(all(x in gset for l in lists for x in l), "mọi kết quả là id gallery hợp lệ")
og, oq = dict(zip(g.id, g.object)), dict(zip(q.id, q.object))
rk = {k: np.mean([any(og.get(x) == oq[i] for x in l[:k]) for i, l in zip(q.id, lists)]) for k in (1, 5)}
B.ok(rk[1] >= NGUONG_R1, f"Recall@1 = {rk[1]:.3f} (ngưỡng {NGUONG_R1}), Recall@5 = {rk[5]:.3f}")
B.ket_thuc()
