"""Chấm bài 02.  python check_02.py [bai_a.py] [bai_b.py] [--khong-chay]
A: truy hồi + MBR (submission_a.csv: id,caption). B: encoder CNN + decoder sinh câu (submission_b.csv). Metric: CIDEr-D.
Chỉ đưa một file thì chỉ chấm phần A (dùng cho bài viết lại nhanh)."""
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, kiem_tra_code, chay, doc_csv  # noqa: E402

NG_A, NG_B = 3.6, 4.5   # mẫu: A 4.27 (truy hồi bằng pixel thu nhỏ: 2.63), B 4.81 (B phải thắng truy hồi)


def tu(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s).split()


def cider_d(hyps, refs, sigma=6.0):
    dem = lambda t: Counter(tuple(t[i:i + k]) for k in range(1, 5) for i in range(len(t) - k + 1))
    rc = [[dem(tu(r)) for r in rs] for rs in refs]
    df = Counter(g for cs in rc for g in set().union(*cs)); log_n = math.log(len(refs))

    def vec(c):
        v, n, dai = [{} for _ in range(4)], [0.0] * 4, 0
        for g, tf in c.items():
            k = len(g) - 1; v[k][g] = tf * (log_n - math.log(max(1.0, df[g]))); n[k] += v[k][g] ** 2
            dai += tf if k == 1 else 0
        return v, [math.sqrt(x) for x in n], dai
    out = []
    for h, cs in zip(hyps, rc):
        vh, nh, lh = vec(dem(tu(h))); s = np.zeros(4)
        for c in cs:
            vr, nr, lr = vec(c)
            for k in range(4):
                val = sum(min(x, vr[k].get(g, 0)) * vr[k].get(g, 0) for g, x in vh[k].items())
                s[k] += (val / (nh[k] * nr[k]) if nh[k] and nr[k] else val) * math.exp(-((lh - lr) ** 2) / (2 * sigma ** 2))
        out.append(s.mean() / len(cs) * 10)
    return float(np.mean(out))


args = [a for a in sys.argv[1:] if not a.startswith("--")] or ["bai_a.py", "bai_b.py"]
files = [Path(a).resolve() for a in args[:2]]
can_chay = "--khong-chay" not in sys.argv
B = Bang("Bài 02 - caption")
gold = json.load(open(HERE / ".dap_an/test_captions.json", encoding="utf-8")); ids = sorted(gold)

for ten, f, ng, sub, kw in [("A - truy hồi + MBR", files[0], NG_A, "submission_a.csv", [(r"argsort|topk", "tìm K láng giềng"), (r"mbr|đồng thuận|dong_thuan", "chọn câu đồng thuận (MBR)")]),
                            ("B - sinh câu", files[1] if len(files) > 1 else None, NG_B, "submission_b.csv", [(r"GRU|LSTM|Transformer", "có decoder GRU / LSTM / Transformer"), (r"ignore_index", "loss bỏ qua token đệm")])]:
    if f is None:
        continue
    print(f"\n{ten}")
    kiem_tra_code(B, f, nfc=True, tu_khoa=kw)
    if can_chay: chay(B, f, timeout_phut=10)
    d = doc_csv(B, f.parent / sub, ["id", "caption"], ids)
    c = cider_d([d.caption[i] for i in ids], [gold[i] for i in ids])
    B.ok(c >= ng, f"{ten[0]}: CIDEr-D = {c:.3f} (ngưỡng {ng})")
B.ket_thuc()
