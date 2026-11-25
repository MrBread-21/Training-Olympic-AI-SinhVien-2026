"""Lời giải ViQuAD_ExtractiveQA (không pretrained): chọn 2 câu theo IDF của từ câu hỏi, sinh span 1-8 âm tiết, đặc trưng span,
XGBoost hồi quy F1 của span, ngưỡng để trống chọn trên dev. Train trên mẫu con 3.000 câu hỏi (đủ RAM, nhanh).
Chạy: python solve.py <thư_mục_dữ_liệu> (và private_test.csv nếu đã giải nén)."""
import math, random, re, sys, time, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

random.seed(42); np.random.seed(42)
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
t0 = time.time()
nfc = lambda s: unicodedata.normalize("NFC", str(s))
tu = lambda s: re.findall(r"\w+", nfc(s).lower())
LOAI = ["bao nhiêu", "mấy", "khi nào", "năm nào", "bao giờ", "thời gian", "ai", "người nào", "ở đâu", "nơi nào", "đâu",
        "tại sao", "vì sao", "như thế nào", "ra sao", "gì", "nào", "là gì", "bao lâu", "thế kỷ", "tên", "số lượng", "phần trăm"]


def f1(p, g):
    p, g = tu(p), tu(g)
    if not p or not g: return float(p == g)
    c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))


tr = pd.read_csv(D / "train.csv", keep_default_na=False); dev = pd.read_csv(D / "dev.csv", keep_default_na=False)
te = pd.read_csv(D / "test.csv", keep_default_na=False)
for p in [Path("private_test.csv"), D / "private_test.csv"]:
    if p.exists(): te = pd.concat([te, pd.read_csv(p, keep_default_na=False)], ignore_index=True); break
cau_tat_ca = [c for ctx in pd.concat([tr.context, dev.context, te.context]).unique() for c in re.split(r"(?<=[.!?;])\s+", nfc(ctx))]
DF = Counter(w for c in cau_tat_ca for w in set(tu(c))); N = len(cau_tat_ca)
idf = lambda w: math.log(1 + N / (1 + DF[w]))


def ung_vien(ctx, q, Lmax=6):
    ctx = nfc(ctx); qs, ql = set(tu(q)), nfc(q).lower()
    co = {k: int(k in ql) for k in LOAI}
    caus = [(m.start(), m.group()) for m in re.finditer(r"[^.!?;]+[.!?;]?", ctx)]
    diem = [sum(idf(w) for w in qs & set(tu(c))) for _, c in caus]
    thu_tu = np.argsort(diem)[::-1]; out = []
    for hang, ci in enumerate(thu_tu[:2]):
        bd, cau = caus[ci]; toks = list(re.finditer(r"\w+", cau)); w = [t.group().lower() for t in toks]
        vt_q = [k for k, x in enumerate(w) if x in qs]
        for i in range(len(toks)):
            for j in range(i + 1, min(i + Lmax, len(toks)) + 1):
                sp = w[i:j]
                if all(x in qs for x in sp): continue
                trai = w[i - 1] if i else "<đầu>"; phai = w[j] if j < len(w) else "<cuối>"
                f = dict(hang=hang, diem_cau=diem[ci], diem_tuong_doi=diem[ci] / (max(diem) + 1e-6), dai=j - i,
                         so=sum(x.isdigit() for x in sp), hoa=int(toks[i].group()[0].isupper()),
                         hoa_het=int(all(t.group()[0].isupper() for t in toks[i:j])), trung=sum(x in qs for x in sp) / (j - i),
                         kc_trai=min([i - k for k in vt_q if k < i] or [30]), kc_phai=min([k - j + 1 for k in vt_q if k >= j] or [30]),
                         q_cua_so=sum(idf(x) for x in w[max(0, i - 6):i] + w[j:j + 6] if x in qs),
                         trai_q=int(trai in qs), phai_q=int(phai in qs), nam=int("năm" in sp or trai == "năm"),
                         **{f"t={trai}": 1, f"p={phai}": 1}, **{f"q:{k}": v for k, v in co.items()})
                for k, v in co.items():
                    if v: f[f"{k}|so"] = int(any(x.isdigit() for x in sp)); f[f"{k}|hoa"] = f["hoa"]; f[f"{k}|t={trai}"] = 1
                out.append((ctx[bd + toks[i].start():bd + toks[j - 1].end()], f))
    return out


def tao(df, co_nhan):
    S, F, G, Y = [], [], [], []
    for qi, r in enumerate(df.itertuples()):
        for s, f in ung_vien(r.context, r.question):
            S.append(s); F.append(f); G.append(qi)
            if co_nhan: Y.append(f1(s, r.answer) if r.answer else 0.0)
    return S, F, np.array(G), np.array(Y)


def tot_nhat(df, model, dv, khoi=400):
    """Span điểm cao nhất của từng câu hỏi, xử lý theo khối để vừa RAM. Trả (chuỗi, điểm)."""
    S_all, P_all = [], []
    for b in range(0, len(df), khoi):
        S, F, G, _ = tao(df.iloc[b:b + khoi], False)
        P = model.predict(dv.transform(F)) if S else np.array([])
        best = pd.DataFrame({"g": G, "p": P, "s": S}).sort_values("p").groupby("g").tail(1).set_index("g")
        for i in range(min(khoi, len(df) - b)):
            S_all.append(best.s[i] if i in best.index else ""); P_all.append(best.p[i] if i in best.index else -1.0)
    return S_all, np.array(P_all)


mau = tr.sample(3000, random_state=42).reset_index(drop=True)
S1, F1, G1, Y1 = tao(mau, True); dv = DictVectorizer(); X1 = dv.fit_transform(F1)
print(f"train: {len(S1)} span ứng viên, {X1.shape[1]} đặc trưng, {time.time() - t0:.0f}s")
m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, subsample=0.8, colsample_bytree=0.6, tree_method="hist", random_state=42).fit(X1, Y1)
del S1, F1, X1
S2, P2 = tot_nhat(dev, m, dv)
diem = {t: np.mean([f1(s if p >= t else "", g) for s, p, g in zip(S2, P2, dev.answer)]) for t in np.arange(0, 0.6, 0.025)}
tot = max(diem, key=diem.get)
print(f"dev F1 = {diem[tot]:.4f} với ngưỡng để trống {tot:.3f} (để trống hết: {np.mean([f1('', g) for g in dev.answer]):.4f}), {time.time() - t0:.0f}s")
S3, P3 = tot_nhat(te, m, dv)
pd.DataFrame({"id": te.id, "answer": [s if p >= tot else "" for s, p in zip(S3, P3)]}).to_csv("submission.csv", index=False)
print(f"đã ghi submission.csv: {len(te)} câu hỏi, {time.time() - t0:.0f}s")
