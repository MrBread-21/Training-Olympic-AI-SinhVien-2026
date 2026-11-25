"""Chấm bài 04.  python check_04.py [bai_a.py] [bai_b.py] [--khong-chay]
A: hồi quy VA cho khía cạnh cho trước (pred_a.jsonl), RMSE_VA. B: trích bộ tứ (pred_b.jsonl), cF1.
Chỉ đưa một file thì chỉ chấm phần A (dùng cho bài viết lại nhanh)."""
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, kiem_tra_code, chay  # noqa: E402

NG_A, NG_B = 0.70, 0.70   # mẫu: A 0.50 (cửa sổ ±4 từ thay vì mệnh đề: 0.95; đoán trung bình: 2.6); B 0.79 (tra từ điển train: 0.57)
args = [a for a in sys.argv[1:] if not a.startswith("--")] or ["bai_a.py", "bai_b.py"]
files = [Path(a).resolve() for a in args[:2]]
can_chay = "--khong-chay" not in sys.argv
B = Bang("Bài 04 - DimABSA")
doc = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def doc_nop(B, p, khoa):
    if not B.ok(p.exists(), f"có file {p.name}"):
        B.ket_thuc()
    try:
        rows = doc(p)
    except json.JSONDecodeError as e:
        B.ok(False, f"{p.name}: mỗi dòng phải là một JSON ({e})"); B.ket_thuc()
    B.ok(all("id" in r and khoa in r for r in rows), f"{p.name}: mỗi dòng có 'id' và '{khoa}'")
    return {r["id"]: r[khoa] for r in rows}


print("A - hồi quy VA")
kiem_tra_code(B, files[0], torch=False, nfc=True, tu_khoa=[(r"clip|np\.clip|min\(9", "kẹp dự đoán vào [1, 9]"),
                                                           (r"ko|teencode|TEEN", "chuẩn hoá teencode")])
if can_chay: chay(B, files[0], timeout_phut=5)
gold = doc(HERE / ".dap_an/a_test.jsonl"); pred = doc_nop(B, files[0].parent / "pred_a.jsonl", "aspects")
B.ok(set(pred) == {r["id"] for r in gold}, f"pred_a.jsonl: đủ {len(gold)} câu")
e, hl = [], True
for r in gold:
    p = {a.get("aspect"): a for a in pred.get(r["id"], [])}
    for g in r["aspects"]:
        try:
            v, a = float(p[g["aspect"]]["valence"]), float(p[g["aspect"]]["arousal"])
            ok = 1 <= v <= 9 and 1 <= a <= 9 and round(v, 2) == v and round(a, 2) == a
        except (KeyError, TypeError, ValueError):
            ok = False
        hl &= ok
        e.append((v - g["valence"]) ** 2 + (a - g["arousal"]) ** 2 if ok else max(g["valence"] - 1, 9 - g["valence"]) ** 2 + max(g["arousal"] - 1, 9 - g["arousal"]) ** 2)
B.ok(hl, "mọi khía cạnh có dự đoán, trong [1, 9], làm tròn 2 chữ số (sai thì tính sai số lớn nhất)", canh_bao=True)
rm = math.sqrt(np.mean(e))
B.ok(rm <= NG_A, f"A: RMSE_VA = {rm:.4f} (ngưỡng ≤ {NG_A})")

if len(files) > 1:
    print("\nB - trích bộ tứ")
    kiem_tra_code(B, files[1], nfc=True, tu_khoa=[(r"B-|BIO", "gán nhãn BIO"), (r"NULL", "xử lý aspect ẩn NULL")])
    if can_chay: chay(B, files[1], timeout_phut=10)
    gold = doc(HERE / ".dap_an/b_test.jsonl"); pred = doc_nop(B, files[1].parent / "pred_b.jsonl", "quadruplets")
    ctp = tp = fp = fn = 0.0; trung = 0
    for r in gold:
        P = pred.get(r["id"], [])
        khoa = lambda q: (str(q.get("aspect", "")).lower(), str(q.get("category", "")).lower(), str(q.get("opinion", "")).lower())
        trung += len(P) - len({khoa(q) for q in P}); khop = 0
        for g in r["quadruplets"]:
            d = []
            for q in P:
                if khoa(q) == khoa(g):
                    try:
                        v, a = float(q["valence"]), float(q["arousal"])
                        d.append(max(0.0, 1 - math.hypot(v - g["valence"], a - g["arousal"]) / math.sqrt(128)) if 1 <= v <= 9 and 1 <= a <= 9 else 0.0)
                    except (KeyError, TypeError, ValueError):
                        d.append(0.0)
            if len(d) == 1: ctp += d[0]; tp += 1; khop += 1
            else: fn += 1
        fp += len(P) - khop
    p_, r_ = ctp / max(tp + fp, 1), ctp / max(tp + fn, 1)
    cf1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
    B.ok(trung == 0, f"không có bộ tứ trùng lặp (đang có {trung})", canh_bao=True)
    B.ok(cf1 >= NG_B, f"B: cF1 = {cf1:.4f} (ngưỡng {NG_B}; cPrecision {p_:.3f}, cRecall {r_:.3f})")
B.ket_thuc()
