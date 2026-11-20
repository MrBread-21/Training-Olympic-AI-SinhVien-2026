"""Chấm bài 29.  python check_29.py [bai_lam.py] [--khong-chay]   (BLEU-4 corpus, chrF; tách từ theo khoảng trắng)"""
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung"))
from cham import Bang, doi_so, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402

NGUONG_BLEU = 0.90   # khung đúng đạt ~0.99 sau 10 epoch; sai mặt nạ hoặc lệch dịch chuỗi đích sẽ tụt hẳn


def bleu(hyp, ref, N=4):
    lp = 0.0
    for n in range(1, N + 1):
        g = lambda t: Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
        hit = sum(sum((g(h) & g(r)).values()) for h, r in zip(hyp, ref)); tot = sum(max(len(h) - n + 1, 0) for h in hyp)
        lp += math.log(max(hit, 1e-9) / max(tot, 1)) / N
    lh, lr = sum(map(len, hyp)), sum(map(len, ref))
    return math.exp(min(0.0, 1 - lr / max(lh, 1)) + lp)


def chrf(hyp, ref, N=6, beta=2):
    P = R = 0.0
    for n in range(1, N + 1):
        g = lambda s: Counter(s[i:i + n] for i in range(len(s) - n + 1))
        hit = tp = tr = 0
        for h, r in zip(hyp, ref):
            gh, gr = g(" ".join(h)), g(" ".join(r)); hit += sum((gh & gr).values()); tp += sum(gh.values()); tr += sum(gr.values())
        P += hit / max(tp, 1) / N; R += hit / max(tr, 1) / N
    return (1 + beta ** 2) * P * R / max(beta ** 2 * P + R, 1e-9)


B = Bang("Bài 29 - seq2seq Transformer")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, tu_khoa=[(r"Transformer|MultiheadAttention", "có Transformer (nn.Transformer hoặc tự viết attention)"),
                               (r"mask", "có mặt nạ nhân quả / padding"),
                               (r"beam", "có beam search"),
                               (r"bleu|BLEU", "tự đo BLEU trên val")])
if can_chay:
    chay(B, bai, timeout_phut=25)
ans = doc_dap_an(HERE, "test.csv")
df = doc_csv(B, bai.parent / "submission.csv", ["id", "vi"], ans.id)
tk = lambda s: unicodedata.normalize("NFC", str(s)).lower().split()
hyp, ref = [tk(s) for s in df.vi], [tk(s) for s in ans.vi]
b, c = bleu(hyp, ref), chrf(hyp, ref)
B.ok(b >= NGUONG_BLEU, f"BLEU-4 = {b:.4f} (ngưỡng {NGUONG_BLEU}), chrF = {c:.4f}, khớp nguyên câu {sum(h == r for h, r in zip(hyp, ref)) / len(ref):.3f}")
B.ket_thuc()
