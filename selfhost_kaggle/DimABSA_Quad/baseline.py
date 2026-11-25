# %% [markdown]
# # Baseline - DimABSA Subtask 2: trích bộ tứ (aspect, category, opinion, valence, arousal)
#
# Không học mô hình, chỉ dùng từ điển rút từ train:
# - khía cạnh: các cụm aspect đã gặp trong train, tìm trong câu (ưu tiên cụm dài);
# - category: nhãn hay gặp nhất của khía cạnh đó trong train;
# - opinion: cụm opinion đã gặp trong train nằm gần khía cạnh nhất;
# - valence, arousal: trung bình VA của opinion đó trong train.
# Bỏ hẳn bộ tứ NULL (dev và test không có NULL). Tự chấm cF1 trên dev, rồi ghi `pred_test.jsonl` (đúng định dạng đề)
# và `submission.csv` (định dạng nộp Kaggle: `id,quadruplets`, cột quadruplets là chuỗi JSON).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("dev.jsonl")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)


def doc(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def hoc(rows, min_dem=2):
    cat, va, dem_asp, dem_op = defaultdict(Counter), defaultdict(list), Counter(), Counter()
    for r in rows:
        for q in r["quadruplets"]:
            a, o = q["aspect"].lower(), q["opinion"].lower()
            if a != "null":
                cat[a][q["category"]] += 1; dem_asp[a] += 1
            if o != "null":
                va[o].append((q["valence"], q["arousal"])); dem_op[o] += 1
    asp = sorted([a for a, c in dem_asp.items() if c >= min_dem], key=len, reverse=True)
    op = sorted([o for o, c in dem_op.items() if c >= min_dem], key=len, reverse=True)
    return asp, op, {a: cat[a].most_common(1)[0][0] for a in asp}, {o: np.mean(va[o], 0) for o in op}


def tim_cum(text, cum):
    """Vị trí (đầu, cuối) của các cụm trong câu, khớp nguyên từ, cụm dài được chọn trước và không chồng nhau."""
    t, dung, out = text.lower(), [False] * (len(text) + 1), []
    for c in cum:
        for m in re.finditer(r"(?<!\w)" + re.escape(c) + r"(?!\w)", t):
            if not any(dung[m.start():m.end()]):
                out.append((m.start(), m.end()))
                for k in range(m.start(), m.end()):
                    dung[k] = True
    return out


def du_doan(r, tu_dien):
    asp, op, cat, va = tu_dien
    A, O = tim_cum(r["text"], asp), tim_cum(r["text"], op)
    out = []
    for a0, a1 in A:
        if not O:
            break
        o0, o1 = min(O, key=lambda o: abs(o[0] - a0))
        if abs(o0 - a0) > 60:
            continue
        v, ar = va[r["text"][o0:o1].lower()]
        out.append({"aspect": r["text"][a0:a1], "category": cat[r["text"][a0:a1].lower()], "opinion": r["text"][o0:o1],
                    "valence": round(float(v), 2), "arousal": round(float(ar), 2)})
    return out


def cf1(gold_rows, pred):
    """Giống metric của cuộc thi (và bộ chấm chính thức DimABSA)."""
    k = lambda q: (q["aspect"].lower(), q["category"].lower(), q["opinion"].lower())
    ctp = tp = fp = fn = 0
    for r in gold_rows:
        P, khop = pred.get(r["id"], []), 0
        for g in r["quadruplets"]:
            d = [max(0, 1 - math.hypot(p["valence"] - g["valence"], p["arousal"] - g["arousal"]) / math.sqrt(128)) for p in P if k(p) == k(g)]
            if len(d) == 1:
                ctp += d[0]; tp += 1; khop += 1
            else:
                fn += 1
        fp += len(P) - khop
    p, rc = ctp / max(tp + fp, 1), ctp / max(tp + fn, 1)
    return 2 * p * rc / (p + rc) if p + rc else 0.0


train, dev = doc(D / "train.jsonl"), doc(D / "dev.jsonl")

# %% [markdown]
# ## Tự chấm trên dev

# %%
td = hoc(train)
print(f"từ điển: {len(td[0])} khía cạnh, {len(td[1])} opinion")
print(f"dev cF1 = {cf1(dev, {r['id']: du_doan(r, td) for r in dev}):.4f}")

# %% [markdown]
# ## Dự đoán test (public) và private_test (nếu đã giải nén), từ điển lấy từ train + dev

# %%
test = doc(D / "test.jsonl")
for thu in [Path("private_test.jsonl"), D / "private_test.jsonl"]:
    if thu.exists():
        test += doc(thu); break
td = hoc(train + dev)
out = [{"id": r["id"], "quadruplets": du_doan(r, td)} for r in test]
with open("pred_test.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
pd.DataFrame({"id": [o["id"] for o in out], "quadruplets": [json.dumps(o["quadruplets"], ensure_ascii=False) for o in out]}).to_csv("submission.csv", index=False)
print(f"ghi pred_test.jsonl và submission.csv: {len(out)} câu, {sum(len(o['quadruplets']) for o in out)} bộ tứ")
