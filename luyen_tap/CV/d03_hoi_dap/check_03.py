"""Chấm bài 03.  python check_03.py [bai_lam.py] [--khong-chay]
Hỏi đáp trích span (submission.csv: id,answer; để trống nếu không có đáp án). Metric: F1 theo từ."""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, kiem_tra_code, chay, doc_dap_an, doi_so  # noqa: E402

NG = 0.80   # lời giải mẫu 0.88; quên NFC 0.60; để trống hết 0.20; trả cả câu chứa nhiều từ câu hỏi nhất 0.12
tu = lambda s: re.findall(r"\w+", unicodedata.normalize("NFC", str(s)).lower())


def f1(p, g):
    p, g = tu(p), tu(g)
    if not p or not g:
        return float(p == g)
    c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))


path, can_chay = doi_so()
B = Bang("Bài 03 - hỏi đáp trích span")
kiem_tra_code(B, path, torch=False, nfc=True, tu_khoa=[(r"GroupShuffleSplit|GroupKFold|context_id", "val chia theo đoạn văn"),
                                                        (r"nguong|threshold", "có ngưỡng để trống")])
if can_chay: chay(B, path, timeout_phut=10)
ans = doc_dap_an(HERE, "test.csv")
sub = pd.read_csv(path.parent / "submission.csv", dtype={"id": str}, keep_default_na=False)
B.ok(list(sub.columns) == ["id", "answer"], f"submission.csv: tiêu đề đúng ['id', 'answer'] (đang là {list(sub.columns)})")
B.ok(sub.id.is_unique and set(sub.id) == set(ans.id), f"submission.csv: đủ {len(ans)} id, mỗi id một dòng (ô answer rỗng được phép)")
p = sub.drop_duplicates("id").set_index("id").answer.reindex(ans.id).fillna("")
diem = np.array([f1(a, b) for a, b in zip(p, ans.answer)])
co = (ans.answer != "").values
B.ok(diem.mean() >= NG, f"F1 = {diem.mean():.4f} (ngưỡng {NG}); câu có đáp án {diem[co].mean():.3f}, câu không đáp án {diem[~co].mean():.3f}")
B.ket_thuc()
