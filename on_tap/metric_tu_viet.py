"""Các metric hay gặp, tự viết bằng numpy / thư viện chuẩn (phòng thi không có `evaluate`, có thể thiếu thư viện).
Chạy `python metric_tu_viet.py` để tự kiểm tra với sklearn (nếu có).

Danh sách: accuracy, macro_f1, auroc, average_precision, rmse, rmse_va, dice, iou, recall_at_k, mrr_at_k,
jaccard_tu, f1_tu, exact_match, bleu4, chrf, cider_d, entity_f1, tiou, continuous_f1, diem_quy_doi."""
import math
import re
import unicodedata
from collections import Counter

import numpy as np


# ------------------------------------------------------------------ phân loại
def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def macro_f1(y, p, lop=None):
    """Trung bình F1 từng lớp, không trọng số. lop: danh sách lớp tính điểm (mặc định: các lớp có trong y)."""
    y, p = np.asarray(y), np.asarray(p)
    lop = sorted(set(y)) if lop is None else lop
    f = []
    for c in lop:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def auroc(y, s):
    """Xác suất mẫu dương có điểm cao hơn mẫu âm (điểm bằng nhau tính 1/2). Dùng hạng trung bình (Mann-Whitney)."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    thu_tu = np.argsort(s, kind="mergesort")
    hang = np.empty(len(s)); hang[thu_tu] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # hạng trung bình cho các điểm bằng nhau
        m = s == v
        if m.sum() > 1:
            hang[m] = hang[m].mean()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((hang[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    """AP = trung bình precision tại vị trí của mỗi mẫu dương khi xếp điểm giảm dần (giống sklearn khi không có điểm bằng nhau)."""
    y = np.asarray(y)[np.argsort(-np.asarray(s, float), kind="mergesort")]
    tp = np.cumsum(y)
    return float(np.sum((tp / np.arange(1, len(y) + 1))[y == 1]) / max(1, y.sum()))


# ------------------------------------------------------------------ hồi quy
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def rmse_va(vg, ag, vp, ap):
    """DimABSA: sqrt(1/N tổng[(Vp-Vg)^2 + (Ap-Ag)^2]) - chia cho N cặp, KHÔNG chia 2N."""
    vg, ag, vp, ap = map(lambda a: np.asarray(a, float), (vg, ag, vp, ap))
    return float(np.sqrt(np.mean((vp - vg) ** 2 + (ap - ag) ** 2)))


# ------------------------------------------------------------------ phân đoạn
def dice(gt, pr, eps=1e-7):
    gt, pr = np.asarray(gt, bool), np.asarray(pr, bool)
    if gt.sum() + pr.sum() == 0:
        return 1.0  # cả hai rỗng: quy ước của đa số cuộc thi
    return float(2 * (gt & pr).sum() / (gt.sum() + pr.sum() + eps))


def iou(gt, pr):
    gt, pr = np.asarray(gt, bool), np.asarray(pr, bool)
    u = (gt | pr).sum()
    return 1.0 if u == 0 else float((gt & pr).sum() / u)


# ------------------------------------------------------------------ truy xuất
def recall_at_k(dung, xep_hang, k):
    """dung: list tập id đúng của mỗi truy vấn; xep_hang: list danh sách id dự đoán (tốt nhất trước)."""
    return float(np.mean([bool(set(r[:k]) & set(d)) for d, r in zip(dung, xep_hang)]))


def mrr_at_k(dung, xep_hang, k=10):
    out = []
    for d, r in zip(dung, xep_hang):
        vt = [i for i, x in enumerate(r[:k]) if x in d]
        out.append(1 / (vt[0] + 1) if vt else 0.0)
    return float(np.mean(out))


# ------------------------------------------------------------------ văn bản: so khớp từ
def chuan_hoa(s, bo_dau_cau=True):
    s = unicodedata.normalize("NFC", str(s)).lower()
    if bo_dau_cau:
        s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return s.split()


def jaccard_tu(a, b):
    a, b = set(chuan_hoa(a)), set(chuan_hoa(b))
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def f1_tu(du_doan, dap_an):
    """F1 kiểu SQuAD: đếm từ trùng có lặp. Cả hai rỗng = 1, một bên rỗng = 0."""
    p, g = chuan_hoa(du_doan), chuan_hoa(dap_an)
    if not p or not g:
        return float(p == g)
    c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))


def exact_match(du_doan, dap_an):
    return float(chuan_hoa(du_doan) == chuan_hoa(dap_an))


# ------------------------------------------------------------------ văn bản: sinh chuỗi
def bleu4(hyps, refs):
    """BLEU-4 mức corpus (giống sacrebleu khi đã tách từ sẵn, không làm mượt). hyps: list chuỗi; refs: list list chuỗi."""
    khop, tong, dai_h, dai_r = [0] * 4, [0] * 4, 0, 0
    for h, rs in zip(hyps, refs):
        h = h.split(); rs = [r.split() for r in rs]
        dai_h += len(h)
        dai_r += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]  # độ dài tham chiếu gần nhất
        for n in range(1, 5):
            ch = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            tran = Counter()
            for r in rs:
                tran |= Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            khop[n - 1] += sum(min(c, tran[g]) for g, c in ch.items()); tong[n - 1] += max(0, len(h) - n + 1)
    if min(khop) == 0:
        return 0.0
    log_p = sum(math.log(k / t) for k, t in zip(khop, tong)) / 4
    bp = 1.0 if dai_h > dai_r else math.exp(1 - dai_r / max(1, dai_h))
    return 100 * bp * math.exp(log_p)


def chrf(hyp, ref, n=6, beta=2):
    """chrF mức câu: F-beta trên n-gram ký tự 1..n (bỏ khoảng trắng), beta = 2 ưu tiên recall."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    p_list, r_list = [], []
    for k in range(1, n + 1):
        ch, cr = Counter(h[i:i + k] for i in range(len(h) - k + 1)), Counter(r[i:i + k] for i in range(len(r) - k + 1))
        m = sum((ch & cr).values())
        if sum(ch.values()) and sum(cr.values()):
            p_list.append(m / sum(ch.values())); r_list.append(m / sum(cr.values()))
    if not p_list:
        return 0.0
    p, rc = np.mean(p_list), np.mean(r_list)
    return 0.0 if p + rc == 0 else float(100 * (1 + beta ** 2) * p * rc / (beta ** 2 * p + rc))


def cider_d(hyps, refs, sigma=6.0):
    """CIDEr-D như pycocoevalcap. hyps: list chuỗi; refs: list list chuỗi. Tách từ: chuan_hoa()."""
    dem = lambda t: Counter(tuple(t[i:i + k]) for k in range(1, 5) for i in range(len(t) - k + 1))
    rc = [[dem(chuan_hoa(r)) for r in rs] for rs in refs]
    df = Counter(g for cs in rc for g in set().union(*cs))
    log_n = math.log(len(refs))

    def vec(c):
        v, nrm, dai = [{} for _ in range(4)], [0.0] * 4, 0
        for g, tf in c.items():
            k = len(g) - 1
            v[k][g] = tf * (log_n - math.log(max(1.0, df[g]))); nrm[k] += v[k][g] ** 2
            dai += tf if k == 1 else 0  # giống mã gốc: độ dài = số bigram
        return v, [math.sqrt(x) for x in nrm], dai

    out = []
    for h, cs in zip(hyps, rc):
        vh, nh, lh = vec(dem(chuan_hoa(h))); s = np.zeros(4)
        for c in cs:
            vr, nr, lr = vec(c)
            for k in range(4):
                val = sum(min(x, vr[k].get(g, 0)) * vr[k].get(g, 0) for g, x in vh[k].items())
                s[k] += (val / (nh[k] * nr[k]) if nh[k] and nr[k] else val) * math.exp(-((lh - lr) ** 2) / (2 * sigma ** 2))
        out.append(s.mean() / len(cs) * 10)
    return float(np.mean(out))


# ------------------------------------------------------------------ gán nhãn chuỗi
def _thuc_the(tags):
    """BIO -> tập (loại, đầu, cuối). I- không có B- đứng trước cũng mở thực thể mới (giống seqeval mặc định)."""
    out, bd, loai = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != loai):
            if loai is not None:
                out.add((loai, bd, i - 1))
            bd, loai = (i, t[2:]) if t != "O" else (None, None)
    return out


def entity_f1(gold_seqs, pred_seqs):
    """Micro-F1 mức thực thể: đúng khi trùng cả loại, vị trí đầu và cuối."""
    tp = fp = fn = 0
    for g, p in zip(gold_seqs, pred_seqs):
        G, P = _thuc_the(g), _thuc_the(p)
        tp += len(G & P); fp += len(P - G); fn += len(G - P)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


# ------------------------------------------------------------------ video, cấu trúc
def tiou(a, b):
    """IoU theo thời gian của hai đoạn [đầu, cuối] (khung, tính cả hai đầu)."""
    giao = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    hop = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - giao
    return giao / hop if hop > 0 else 0.0


def continuous_f1(gold, pred):
    """DimABSA cF1. gold, pred: list (mỗi câu) list bộ (aspect, category, opinion, V, A)."""
    ctp = tp = fp = fn = 0.0
    for G, P in zip(gold, pred):
        khoa = lambda q: (q[0].lower(), q[1].lower(), q[2].lower())
        khop = 0
        for g in G:
            d = [max(0.0, 1 - math.hypot(p[3] - g[3], p[4] - g[4]) / math.sqrt(128)) if 1 <= p[3] <= 9 and 1 <= p[4] <= 9 else 0.0
                 for p in P if khoa(p) == khoa(g)]
            if len(d) == 1:
                ctp += d[0]; tp += 1; khop += 1
            else:
                fn += 1  # không khớp, hoặc khớp nhiều dự đoán trùng nhau
        fp += len(P) - khop
    p, r = ctp / max(tp + fp, 1), ctp / max(tp + fn, 1)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ------------------------------------------------------------------ điểm OlpAI
def diem_quy_doi(diem, min_score, max_score, cao_la_tot=True):
    """(điểm - Min) / (Max - Min), kẹp trong [0, 1]. Metric thấp-là-tốt (RMSE) thì đổi chiều."""
    x = (diem - min_score) / (max_score - min_score) if cao_la_tot else (min_score - diem) / (min_score - max_score)
    return float(min(1.0, max(0.0, x)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500); p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 4, 500))
    yb = rng.integers(0, 2, 500); s = yb * 0.5 + rng.random(500); s[:50] = np.round(s[:50], 1)
    try:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
        print("macro_f1", macro_f1(y, p), f1_score(y, p, average="macro"))
        print("auroc   ", auroc(yb, s), roc_auc_score(yb, s))
        s2 = yb * 0.5 + rng.random(500)
        print("AP      ", average_precision(yb, s2), average_precision_score(yb, s2))
    except ImportError:
        print("không có sklearn")
    print("f1_tu   ", f1_tu("tên Lâm Bá Kiệt", "Lâm Bá Kiệt"), "(0.857)")
    print("rmse_va ", rmse_va([7.5, 3.33], [7.67, 5.83], [7, 4], [7, 6]), "(0.7670)")
    print("entity  ", entity_f1([["B-PER", "I-PER", "O", "B-LOC"]], [["B-PER", "I-PER", "O", "B-ORG"]]), "(0.5)")
    print("tiou    ", tiou((10, 19), (15, 24)), "(5/15 = 0.333)")
    print("bleu4   ", bleu4(["the cat sat on the mat"], [["the cat sat on the mat"]]), "(100)")
    print("cider   ", cider_d(["một người đang đi xe máy", "con mèo nằm trên ghế"],
                              [["một người đi xe máy trên đường", "người đàn ông đi xe máy"], ["con mèo nằm ngủ trên ghế", "mèo trên ghế"]]))
    print("cF1     ", continuous_f1([[("pizza", "FOOD#QUALITY", "great", 7.5, 7.67)]], [[("Pizza", "FOOD#QUALITY", "great", 7.0, 7.0)]]))
    print("quy đổi ", diem_quy_doi(1.5, 1.9, 1.1, cao_la_tot=False), "(RMSE 1.5 giữa Min 1.9 và Max 1.1 = 0.5)")
