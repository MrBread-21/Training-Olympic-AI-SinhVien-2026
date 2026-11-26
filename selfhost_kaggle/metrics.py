"""Các hàm chấm theo giao diện metric notebook của Kaggle:
    score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float
Mỗi bài dùng một hàm; file host/metric.py của bài được sinh ra từ đây (phần chung + đúng một hàm score).
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


def _ghep(solution, submission, row_id_column_name, cot):
    sol = solution.drop(columns=["Usage"], errors="ignore").copy()
    sub = submission.copy()
    if row_id_column_name not in sub.columns:
        raise ParticipantVisibleError(f"Thiếu cột {row_id_column_name}")
    for c in cot:
        if c not in sub.columns:
            raise ParticipantVisibleError(f"Thiếu cột {c}")
    if sub[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError("Có id bị lặp trong file nộp")
    sol[row_id_column_name] = sol[row_id_column_name].astype(str)
    sub[row_id_column_name] = sub[row_id_column_name].astype(str)
    thieu = set(sol[row_id_column_name]) - set(sub[row_id_column_name])
    if thieu:
        raise ParticipantVisibleError(f"Thiếu {len(thieu)} id, ví dụ {sorted(thieu)[:3]}")
    return sol.merge(sub[[row_id_column_name] + cot], on=row_id_column_name, how="left", suffixes=("", "_nop"))


def _chuoi(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


# ---------------------------------------------------------------- phân loại
def macro_f1(solution, submission, row_id_column_name):
    """Cột label. Macro-F1 trên các lớp có trong đáp án."""
    m = _ghep(solution, submission, row_id_column_name, ["label"])
    y = m["label"].map(_chuoi).values
    p = m["label_nop"].map(_chuoi).values
    f1 = []
    for c in sorted(set(y)):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def multilabel_macro_f1(solution, submission, row_id_column_name):
    """Cột labels: tên lớp cách nhau bởi dấu cách (có thể rỗng). Macro-F1 trên các lớp có trong đáp án."""
    m = _ghep(solution, submission, row_id_column_name, ["labels"])
    Y = [set(_chuoi(s).split()) for s in m["labels"]]
    P = [set(_chuoi(s).split()) for s in m["labels_nop"]]
    lop = sorted(set().union(*Y))
    f1 = []
    for c in lop:
        tp = sum(c in y and c in p for y, p in zip(Y, P))
        fp = sum(c not in y and c in p for y, p in zip(Y, P))
        fn = sum(c in y and c not in p for y, p in zip(Y, P))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def auroc(solution, submission, row_id_column_name):
    """Cột label (0/1) trong đáp án, cột score (số thực, lớn = bất thường) trong file nộp."""
    m = _ghep(solution, submission, row_id_column_name, ["score"])
    s = pd.to_numeric(m["score"], errors="coerce")
    if s.isna().any():
        raise ParticipantVisibleError("Cột score phải là số, không được rỗng")
    y = m["label"].astype(int).values
    r = s.rank(method="average").values  # Mann-Whitney, xử lý điểm bằng nhau
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- phân đoạn
def rle_decode(rle, h, w):
    """RLE: 'start length start length ...', start đánh số từ 1, duyệt ảnh theo hàng (row-major)."""
    m = np.zeros(h * w, dtype=np.uint8)
    s = _chuoi(rle).split()
    if len(s) % 2:
        raise ParticipantVisibleError("RLE phải có số phần tử chẵn")
    for a, l in zip(s[0::2], s[1::2]):
        a, l = int(a) - 1, int(l)
        if a < 0 or l < 0 or a + l > h * w:
            raise ParticipantVisibleError("RLE vượt ngoài ảnh")
        m[a:a + l] = 1
    return m.reshape(h, w)


def mean_dice(solution, submission, row_id_column_name):
    """Cột rle (đáp án có thêm height, width). Dice trung bình theo ảnh; cả hai rỗng thì Dice = 1."""
    m = _ghep(solution, submission, row_id_column_name, ["rle"])
    d = []
    for _, r in m.iterrows():
        h, w = int(r["height"]), int(r["width"])
        a, b = rle_decode(r["rle"], h, w), rle_decode(r["rle_nop"], h, w)
        s = a.sum() + b.sum()
        d.append(1.0 if s == 0 else 2 * (a & b).sum() / s)
    return float(np.mean(d))


# ---------------------------------------------------------------- truy xuất
def recall_at_1(solution, submission, row_id_column_name):
    """Nordland. Đáp án: positives (id ảnh database đúng, cách nhau dấu cách).
    File nộp: predictions (id ảnh database, xếp từ khớp nhất, tối đa 10). Điểm = Recall@1."""
    return _recall(solution, submission, row_id_column_name, 1)


def _recall(solution, submission, row_id_column_name, k):
    m = _ghep(solution, submission, row_id_column_name, ["predictions"])
    ok = []
    for pos, pred in zip(m["positives"], m["predictions"]):
        p = _chuoi(pred).split()
        if len(p) > 10:
            raise ParticipantVisibleError("Mỗi truy vấn nộp tối đa 10 ảnh")
        ok.append(bool(set(p[:k]) & set(_chuoi(pos).split())))
    return float(np.mean(ok))


def recall_at_1_vpr(solution, submission, row_id_column_name):
    """Định vị tàu hoả bốn mùa (OlpAI'26). Đáp án: id, positives (các winter_id lệch tối đa 10 khung, cách nhau dấu cách).
    File nộp: id, winter_id (đúng một ảnh mùa đông cho mỗi ảnh mùa hè). Điểm = Recall@1 x 100, thang 0-100.
    Luật chấm như đề: id lặp thì chỉ chấm dòng đầu; dòng có id lạ bị bỏ qua; ảnh không có dòng nào
    hoặc winter_id không thuộc kho tính là sai; thiếu cột hoặc sai tên cột được 0 điểm."""
    sol = solution.drop(columns=["Usage"], errors="ignore")
    if row_id_column_name not in submission.columns or "winter_id" not in submission.columns:
        return 0.0
    sub = submission.astype({row_id_column_name: str}).drop_duplicates(row_id_column_name, keep="first")
    du_doan = dict(zip(sub[row_id_column_name], sub["winter_id"].map(_chuoi).str.strip()))
    dung = [du_doan.get(str(q), "") in set(_chuoi(p).split()) for q, p in zip(sol[row_id_column_name], sol["positives"])]
    return float(100 * np.mean(dung))


def mrr_at_10(solution, submission, row_id_column_name):
    """Đáp án: passage_id (một đoạn đúng). File nộp: passage_ids (tối đa 10, xếp từ khớp nhất). MRR@10."""
    m = _ghep(solution, submission, row_id_column_name, ["passage_ids"])
    rr = []
    for g, pred in zip(m["passage_id"].map(_chuoi), m["passage_ids"]):
        p = _chuoi(pred).split()
        if len(p) > 10:
            raise ParticipantVisibleError("Mỗi câu hỏi nộp tối đa 10 đoạn")
        rr.append(1.0 / (p.index(g) + 1) if g in p else 0.0)
    return float(np.mean(rr))


# ---------------------------------------------------------------- văn bản
def entity_f1(solution, submission, row_id_column_name):
    """NER. Cột tags: nhãn BIO cách nhau dấu cách, mỗi token một nhãn. F1 micro theo thực thể (khớp đúng biên và loại)."""
    m = _ghep(solution, submission, row_id_column_name, ["tags"])

    def spans(tags):
        out, st, typ = set(), None, None
        for i, t in enumerate(tags + ["O"]):
            if st is not None and not (t.startswith("I-") and t[2:] == typ):
                out.add((st, i, typ)); st = None
            if t.startswith("B-") or (t.startswith("I-") and st is None):
                st, typ = i, t[2:]
        return out

    tp = fp = fn = 0
    for g, p in zip(m["tags"], m["tags_nop"]):
        g = _chuoi(g).split(); p = _chuoi(p).split()
        p = (p + ["O"] * len(g))[:len(g)]  # thiếu thì coi là O, thừa thì cắt
        G, P = spans(g), spans(p)
        tp += len(G & P); fp += len(P - G); fn += len(G - P)
    return 0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn))


def word_accuracy(solution, submission, row_id_column_name):
    """Thêm dấu. Cột text. Tỉ lệ từ (tách theo dấu cách) đúng vị trí; thiếu từ tính là sai."""
    import unicodedata
    m = _ghep(solution, submission, row_id_column_name, ["text"])
    dung = tong = 0
    for g, p in zip(m["text"], m["text_nop"]):
        g = unicodedata.normalize("NFC", _chuoi(g)).split()
        p = unicodedata.normalize("NFC", _chuoi(p)).split()
        tong += len(g); dung += sum(a == b for a, b in zip(g, p))
    return float(dung / tong)


def _tach(s):
    import unicodedata
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFC", _chuoi(s)).lower())


def bleu4(solution, submission, row_id_column_name):
    """Dịch máy. Cột vi. BLEU-4 mức corpus (chữ thường, tách từ và dấu câu), giống SacreBLEU nhưng thang 0-100."""
    m = _ghep(solution, submission, row_id_column_name, ["vi"])
    khop = [0] * 4; tong = [0] * 4; c = r = 0
    for g, p in zip(m["vi"], m["vi_nop"]):
        g, p = _tach(g), _tach(p)
        c += len(p); r += len(g)
        for n in range(1, 5):
            pg = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
            gg = Counter(tuple(g[i:i + n]) for i in range(len(g) - n + 1))
            khop[n - 1] += sum(min(v, gg[k]) for k, v in pg.items()); tong[n - 1] += max(len(p) - n + 1, 0)
    if c == 0 or min(khop) == 0:
        return 0.0
    bp = 1.0 if c > r else math.exp(1 - r / c)
    return float(100 * bp * math.exp(sum(math.log(k / t) for k, t in zip(khop, tong)) / 4))


# ---------------------------------------------------------------- DimABSA (đề NLP OlpAI'26)
def rmse_va(solution, submission, row_id_column_name):
    """DimABSA Subtask 1 (hồi quy Valence-Arousal cho khía cạnh cho trước). Thấp hơn là tốt hơn.
    Đáp án: id, aspects (JSON: [{"aspect", "valence", "arousal"}]). File nộp: id, aspects (JSON cùng dạng).
    RMSE_VA = sqrt(1/N x tổng [(Vp - Vg)^2 + (Ap - Ag)^2]) trên mọi cặp (câu, khía cạnh) của đáp án.
    Dự đoán khớp với khía cạnh theo đúng chuỗi aspect. Khía cạnh không có dự đoán, giá trị không đọc được
    hoặc nằm ngoài [1, 9] bị tính sai số lớn nhất có thể (xa nhất về biên 1 hoặc 9 trên mỗi chiều)."""
    import json
    sol = solution.drop(columns=["Usage"], errors="ignore")
    for c in (row_id_column_name, "aspects"):
        if c not in submission.columns:
            raise ParticipantVisibleError(f"Thiếu cột {c}")
    nop = {}
    for i, s in zip(submission[row_id_column_name].astype(str), submission["aspects"]):
        if i in nop:
            continue  # id lặp: chỉ lấy dòng đầu
        try:
            ds = json.loads(_chuoi(s) or "[]")
            nop[i] = {str(d["aspect"]): d for d in ds if isinstance(d, dict) and "aspect" in d}
        except (ValueError, TypeError):
            nop[i] = {}
    tong, n = 0.0, 0
    for i, s in zip(sol[row_id_column_name].astype(str), sol["aspects"]):
        p = nop.get(i, {})
        for g in json.loads(s):
            d = 0.0
            for k in ("valence", "arousal"):
                gv = float(g[k])
                try:
                    pv = float(p[g["aspect"]][k])
                    ok = 1.0 <= pv <= 9.0
                except (KeyError, ValueError, TypeError):
                    ok = False
                d += (pv - gv) ** 2 if ok else max(gv - 1.0, 9.0 - gv) ** 2
            tong += d; n += 1
    return float(math.sqrt(tong / n))


def continuous_f1(solution, submission, row_id_column_name):
    """DimABSA Subtask 2 (trích bộ tứ aspect, category, opinion, VA). Điểm cF1 thang 0-1, giống bộ chấm chính thức.
    Đáp án: id, quadruplets (JSON: [{"aspect", "category", "opinion", "valence", "arousal"}]). File nộp: id, quadruplets.
    Một dự đoán khớp nhãn chuẩn khi (aspect, category, opinion) trùng khớp sau khi chuyển chữ thường; khi khớp,
    cTP = max(0, 1 - khoảng cách Euclid VA / sqrt(128)), VA ngoài [1, 9] cho cTP = 0.
    Nhãn chuẩn khớp với nhiều dự đoán trùng nhau: cTP = 0, tính bỏ sót. cPrecision = tổng cTP / (TP + FP),
    cRecall = tổng cTP / (TP + FN), cF1 là trung bình điều hoà. Câu không có dòng nộp coi như không dự đoán gì."""
    import json
    sol = solution.drop(columns=["Usage"], errors="ignore")
    for c in (row_id_column_name, "quadruplets"):
        if c not in submission.columns:
            raise ParticipantVisibleError(f"Thiếu cột {c}")

    def doc(s):
        try:
            ds = json.loads(_chuoi(s) or "[]")
        except ValueError:
            return []
        out = []
        for d in ds if isinstance(ds, list) else []:
            try:
                out.append(((str(d["aspect"]).lower(), str(d["category"]).lower(), str(d["opinion"]).lower()),
                            float(d["valence"]), float(d["arousal"])))
            except (KeyError, ValueError, TypeError):
                out.append((None, 0.0, 0.0))  # bộ tứ hỏng: chỉ tính là dự đoán sai
        return out

    nop = {}
    for i, s in zip(submission[row_id_column_name].astype(str), submission["quadruplets"]):
        nop.setdefault(i, doc(s))
    ctp, tp, fp, fn = 0.0, 0, 0, 0
    for i, s in zip(sol[row_id_column_name].astype(str), sol["quadruplets"]):
        gold, pred = doc(s), nop.get(i, [])
        khop = 0
        for k, gv, ga in gold:
            diem = []
            for kp, pv, pa in pred:
                if kp == k:
                    ok = 1 <= pv <= 9 and 1 <= pa <= 9
                    diem.append(max(0.0, 1 - math.hypot(pv - gv, pa - ga) / math.sqrt(128)) if ok else 0.0)
            if len(diem) == 1:
                khop += 1; tp += 1; ctp += diem[0]
            else:
                fn += 1
        fp += len(pred) - khop
    p = ctp / (tp + fp) if tp + fp else 0.0
    r = ctp / (tp + fn) if tp + fn else 0.0
    return float(2 * p * r / (p + r)) if p + r else 0.0


# ---------------------------------------------------------------- hỏi đáp trích xuất
def qa_f1(solution, submission, row_id_column_name):
    """Hỏi đáp trích xuất (kiểu SQuAD 2.0). Đáp án: answers (JSON: danh sách câu trả lời đúng; [""] nếu câu hỏi không có đáp án).
    File nộp: answer (chuỗi trích từ đoạn văn; để trống nếu cho rằng không có đáp án).
    Chuẩn hoá: Unicode NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng. Mỗi câu hỏi lấy F1 theo từ cao nhất với các đáp án đúng;
    câu không có đáp án được 1 nếu để trống, 0 nếu trả lời. Điểm = trung bình F1, thang 0-1."""
    import json
    import unicodedata
    m = _ghep(solution, submission, row_id_column_name, ["answer"])

    def chuan(s):
        s = unicodedata.normalize("NFC", _chuoi(s)).lower()
        s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
        return s.split()

    diem = []
    for g, p in zip(m["answers"], m["answer"]):
        p = chuan(p)
        best = 0.0
        for a in json.loads(g):
            a = chuan(a)
            if not a or not p:
                best = max(best, float(a == p))
                continue
            chung = sum((Counter(a) & Counter(p)).values())
            if chung:
                pr, rc = chung / len(p), chung / len(a)
                best = max(best, 2 * pr * rc / (pr + rc))
        diem.append(best)
    return float(np.mean(diem))


# ---------------------------------------------------------------- lập kế hoạch đường đi
def path_score(solution, submission, row_id_column_name):
    """Robot đi qua mọi điểm mốc rồi về đích. Đáp án: grid (các hàng nối bằng '/'), optimal_cost.
    File nộp: actions (chuỗi U/D/L/R). Ô: '#' tường, '1'-'9' chi phí khi bước vào, 'S' xuất phát, 'G' đích, 'A'-'L' điểm mốc (chi phí 1).
    Đường hợp lệ khi không ra ngoài lưới, không đi vào tường, đi qua mọi điểm mốc (thứ tự tuỳ ý) và kết thúc ở G.
    Mỗi bản đồ: hợp lệ thì điểm = optimal_cost / chi phí của bạn (tối đa 1), không hợp lệ thì 0. Điểm = trung bình, thang 0-1."""
    m = _ghep(solution, submission, row_id_column_name, ["actions"])
    buoc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    diem = []
    for g, opt, act in zip(m["grid"], m["optimal_cost"], m["actions"]):
        luoi = _chuoi(g).split("/")
        H, W = len(luoi), len(luoi[0])
        act = _chuoi(act).strip().upper()
        if len(act) > 4 * H * W or any(c not in buoc for c in act):
            diem.append(0.0); continue
        y, x = next((i, r.index("S")) for i, r in enumerate(luoi) if "S" in r)
        can = {c for r in luoi for c in r if "A" <= c <= "L"}
        cost, ok = 0, True
        for c in act:
            y, x = y + buoc[c][0], x + buoc[c][1]
            if not (0 <= y < H and 0 <= x < W) or luoi[y][x] == "#":
                ok = False; break
            o = luoi[y][x]
            cost += int(o) if o.isdigit() else 1
            can.discard(o)
        ok = ok and not can and luoi[y][x] == "G"
        diem.append(min(1.0, float(opt) / cost) if ok and cost > 0 else 0.0)
    return float(np.mean(diem))


# ---------------------------------------------------------------- sinh caption
def _cider_toks(s):
    import unicodedata
    s = unicodedata.normalize("NFC", _chuoi(s)).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s).split()


def cider_d(solution, submission, row_id_column_name):
    """Sinh caption. Đáp án: captions (JSON: các câu tham chiếu). File nộp: caption (một câu).
    CIDEr-D như pycocoevalcap: n-gram 1-4, TF-IDF với tần suất văn bản tính trên tham chiếu của tập đang chấm,
    cắt ngưỡng theo tham chiếu, phạt chênh độ dài (sigma = 6), nhân 10. Tách từ: chữ thường, bỏ dấu câu, theo dấu cách."""
    import json
    m = _ghep(solution, submission, row_id_column_name, ["caption"])
    refs = [[_cider_toks(r) for r in json.loads(g)] for g in m["captions"]]
    hyps = [_cider_toks(c) for c in m["caption"]]

    def dem(t):
        return Counter(tuple(t[i:i + k]) for k in range(1, 5) for i in range(len(t) - k + 1))

    rc = [[dem(r) for r in rs] for rs in refs]
    df = Counter(g for cs in rc for g in set().union(*cs))
    log_n = math.log(float(len(refs)))

    def vec(c):
        v, nrm, dai = [{} for _ in range(4)], [0.0] * 4, 0
        for g, tf in c.items():
            k = len(g) - 1
            v[k][g] = tf * (log_n - math.log(max(1.0, df[g])))
            nrm[k] += v[k][g] ** 2
            if k == 1:
                dai += tf  # giống mã gốc: độ dài tính bằng số bigram
        return v, [math.sqrt(x) for x in nrm], dai

    diem = []
    for h, cs in zip(hyps, rc):
        vh, nh, lh = vec(dem(h))
        s = np.zeros(4)
        for c in cs:
            vr, nr, lr = vec(c)
            for k in range(4):
                val = sum(min(x, vr[k].get(g, 0.0)) * vr[k].get(g, 0.0) for g, x in vh[k].items())
                if nh[k] and nr[k]:
                    val /= nh[k] * nr[k]
                s[k] += val * math.exp(-((lh - lr) ** 2) / (2 * 6.0 ** 2))
        diem.append(s.mean() / len(cs) * 10.0)
    return float(np.mean(diem))


# ---------------------------------------------------------------- tấn công đối kháng
_TRONG_SO = {}  # metric.py trên Kaggle: được thay bằng trọng số (base64, float16) của mô hình A và mô hình ẩn B


def _cnn_np(x, w):
    """TinyCNN (3 khối conv3x3 + ReLU + maxpool 2, rồi linear) chạy bằng numpy. x: (N, 32, 32, 3) uint8."""
    from numpy.lib.stride_tricks import sliding_window_view
    if len(x) > 256:
        return np.concatenate([_cnn_np(x[i:i + 256], w) for i in range(0, len(x), 256)])
    h = (x.astype(np.float32) / 255.0 - np.array([0.4914, 0.4822, 0.4465], np.float32)) / np.array([0.2470, 0.2435, 0.2616], np.float32)
    for i in range(3):
        k, b = w[f"c{i}.weight"], w[f"c{i}.bias"]  # (out, in, 3, 3)
        p = np.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)))
        cua = sliding_window_view(p, (3, 3), axis=(1, 2))  # (N, H, W, in, 3, 3)
        h = np.maximum(np.einsum("nhwcij,ocij->nhwo", cua, k, optimize=True) + b, 0)
        n, hh, ww, c = h.shape
        h = h.reshape(n, hh // 2, 2, ww // 2, 2, c).max(axis=(2, 4))
    h = h.transpose(0, 3, 1, 2).reshape(len(h), -1)  # cùng thứ tự flatten với PyTorch (N, C, H, W)
    return (h @ w["fc.weight"].T + w["fc.bias"]).argmax(1)


def adv_success(solution, submission, row_id_column_name):
    """Tấn công đối kháng CIFAR-10. Đáp án: label, orig (base64 của 3072 byte uint8, ảnh 32x32x3 theo HWC).
    File nộp: image (base64 cùng định dạng). Ảnh hợp lệ khi giải mã được đúng 3072 byte và mọi điểm ảnh lệch ảnh gốc
    không quá 8 (thang 0-255). Mỗi ảnh hợp lệ được 0,5 nếu đánh lừa mô hình A (đã công bố) và 0,5 nếu đánh lừa
    mô hình ẩn B; ảnh không hợp lệ được 0. Điểm = trung bình, thang 0-1."""
    import base64
    m = _ghep(solution, submission, row_id_column_name, ["image"])
    w = {ten: {k: np.frombuffer(base64.b64decode(v[0]), np.float16).astype(np.float32).reshape(v[1])
               for k, v in ts.items()} for ten, ts in _TRONG_SO.items()}
    goc = np.stack([np.frombuffer(base64.b64decode(s), np.uint8) for s in m["orig"]]).astype(np.int16)
    anh, hop_le = np.zeros_like(goc), np.zeros(len(m), bool)
    for i, s in enumerate(m["image"]):
        try:
            a = np.frombuffer(base64.b64decode(_chuoi(s), validate=True), np.uint8)
        except ValueError:
            continue
        if a.size == 3072 and np.abs(a.astype(np.int16) - goc[i]).max() <= 8:
            anh[i], hop_le[i] = a, True
    x = anh.astype(np.uint8).reshape(-1, 32, 32, 3)
    y = m["label"].astype(int).values
    diem = 0.5 * (_cnn_np(x, w["A"]) != y) + 0.5 * (_cnn_np(x, w["B"]) != y)
    return float(np.mean(np.where(hop_le, diem, 0.0)))
