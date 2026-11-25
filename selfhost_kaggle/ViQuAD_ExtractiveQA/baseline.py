# %% [markdown]
# # Baseline - Hỏi đáp trích xuất tiếng Việt (ViQuAD 2.0)
#
# Không học mô hình: chọn câu trong đoạn văn chứa nhiều từ của câu hỏi nhất, trả lời cả câu đó nếu nó chứa ít nhất
# 80% số từ của câu hỏi, nếu không thì để trống (không có đáp án). Khoảng 30% câu hỏi không có đáp án, nên chiến lược
# "để trống hết" đã được khoảng 0,3: mọi cách làm thật sự phải trích đúng cụm ngắn để vượt mốc này.
# Tự chấm F1 trên dev, rồi ghi `submission.csv` (`id,answer`).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("dev.csv")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)


def tu(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in s).split()


def tra_loi(context, question, nguong=0.8):
    """Câu trong đoạn văn chứa nhiều từ của câu hỏi nhất; trả lời cả câu đó nếu nó chứa ít nhất `nguong` phần từ của
    câu hỏi, nếu không thì để trống (coi là không có đáp án)."""
    hoi = set(tu(question))
    cau = [c for c in re.split(r"(?<=[.!?;])\s+", str(context)) if c.strip()]
    if not cau or not hoi:
        return ""
    trung = [len(hoi & set(tu(c))) for c in cau]
    i = int(np.argmax(trung))
    return cau[i].strip() if trung[i] / len(hoi) >= nguong else ""


def f1(pred, golds):
    p, best = tu(pred), 0.0
    for g in golds:
        g = tu(g)
        if not g or not p:
            best = max(best, float(g == p)); continue
        c = sum((Counter(g) & Counter(p)).values())
        if c:
            best = max(best, 2 * c / (len(g) + len(p)))
    return best


# %% [markdown]
# ## Tự chấm trên dev (chọn ngưỡng tỉ lệ từ trùng)

# %%
dev = pd.read_csv(D / "dev.csv", keep_default_na=False)
for ng in [0.6, 0.7, 0.8, 0.9]:
    s = np.mean([f1(tra_loi(c, q, ng), [a]) for c, q, a in zip(dev.context, dev.question, dev.answer)])
    print(f"ngưỡng {ng}: dev F1 = {s:.4f}")
print(f"(để trống tất cả: {np.mean([f1('', [a]) for a in dev.answer]):.4f})")

# %% [markdown]
# ## Dự đoán test (public) và private_test (nếu đã giải nén)

# %%
test = pd.read_csv(D / "test.csv", keep_default_na=False)
for thu in [Path("private_test.csv"), D / "private_test.csv"]:
    if thu.exists():
        test = pd.concat([test, pd.read_csv(thu, keep_default_na=False)]); break
pd.DataFrame({"id": test.id, "answer": [tra_loi(c, q, 0.8) for c, q in zip(test.context, test.question)]}).to_csv("submission.csv", index=False)
print(f"ghi submission.csv: {len(test)} câu hỏi")
