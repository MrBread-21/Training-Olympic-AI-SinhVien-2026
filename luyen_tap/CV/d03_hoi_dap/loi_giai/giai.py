import random, re, unicodedata
from collections import Counter
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupShuffleSplit

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
seed_all(42)
nfc = lambda s: unicodedata.normalize("NFC", str(s))
tu = lambda s: re.findall(r"\w+", nfc(s).lower())
LOAI = ["năm nào", "năm sinh", "ở đâu", "tỉnh nào", "nghề", "trường nào", "tác phẩm", "giải", "bao nhiêu",
        "tốt nghiệp", "ra trường", "công bố", "ra mắt", "trao", "nhận", "sinh", "ra đời", "công trình", "học trò"]

def f1(p, g):
    p, g = tu(p), tu(g)
    if not p or not g: return float(p == g)
    c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))

def ung_vien(ctx, q, Lmax=6, so_cau=2):
    """Span 1-Lmax token (theo \w+) trong `so_cau` câu khớp câu hỏi nhất, kèm đặc trưng. Trả list (chuỗi span, dict)."""
    qs, ql = set(tu(q)), nfc(q).lower()
    noi_dung = {t.lower() for t in re.findall(r"\w+", nfc(q)) if t.islower()}     # từ viết thường của câu hỏi (bỏ tên riêng)
    co = {k: int(k in ql) for k in LOAI}
    caus = [m.group() for m in re.finditer(r"[^.]+\.?", ctx)]                      # tách câu
    diem = [len(noi_dung & set(tu(c))) + 0.5 * len(qs & set(tu(c))) for c in caus]
    out = []
    for hang, ci in enumerate(np.argsort(diem)[::-1][:so_cau]):
        cau = caus[ci]; toks = list(re.finditer(r"\w+", cau)); w = [t.group().lower() for t in toks]
        for i in range(len(toks)):
            for j in range(i + 1, min(i + Lmax, len(toks)) + 1):
                sp = w[i:j]
                if all(x in qs for x in sp): continue
                s = cau[toks[i].start():toks[j - 1].end()]
                trai = w[i - 1] if i > 0 else "<đầu>"; trai2 = w[i - 2] if i > 1 else "<đầu>"
                phai = w[j] if j < len(w) else "<cuối>"
                kc = [abs(k - i) for k, x in enumerate(w) if x in noi_dung] or [99]
                f = dict(hang_cau=hang, diem_cau=diem[ci], dai=j - i, so=int(sp[0].isdigit()),
                         nam=int(len(sp) == 1 and sp[0].isdigit() and len(sp[0]) == 4),
                         hoa=int(s[0].isupper()), hoa_het=int(all(t.group()[0].isupper() for t in toks[i:j])),
                         trung_span=sum(x in qs for x in sp), kc=min(kc),
                         ngoac=int(i > 0 and cau[toks[i].start() - 1] == '"'), het_ngoac=int(cau[toks[j - 1].end():toks[j - 1].end() + 1] == '"'),
                         **{f"trai={trai}": 1, f"trai2={trai2}": 1, f"phai={phai}": 1}, **{f"q:{k}": v for k, v in co.items()})
                for k, v in co.items():                       # tương tác loại câu hỏi x từ bên trái
                    if v: f[f"{k}|{trai}"] = 1
                out.append((s, f))
    return out

def tao_mau(df, co_nhan):
    S, F, G, Y = [], [], [], []
    for qi, r in enumerate(df.itertuples()):
        for s, f in ung_vien(nfc(r.context), nfc(r.question)):
            S.append(s); F.append(f); G.append(qi)
            if co_nhan: Y.append(f1(s, r.answer) if r.answer else 0.0)
    return S, F, np.array(G), np.array(Y)

def du_doan(df, S, P, G, nguong):
    best = pd.DataFrame({"g": G, "p": P, "s": S}).sort_values("p").groupby("g").tail(1).set_index("g")
    return [best.s[i] if i in best.index and best.p[i] >= nguong else "" for i in range(len(df))]

tr = pd.read_csv("data/train.csv", keep_default_na=False); te = pd.read_csv("data/test.csv", keep_default_na=False)
a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(tr, groups=tr.context_id))   # val theo đoạn văn
fit, val = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)
dv = DictVectorizer()
S1, F1, G1, Y1 = tao_mau(fit, True); S2, F2, G2, _ = tao_mau(val, False)
X1, X2 = dv.fit_transform(F1), dv.transform(F2)
mo_hinh = lambda: xgb.XGBRegressor(n_estimators=200, learning_rate=0.15, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42)
m = mo_hinh().fit(X1, Y1)
P2 = m.predict(X2)
tot = max(np.arange(0, 0.8, 0.05), key=lambda t: np.mean([f1(p, g) for p, g in zip(du_doan(val, S2, P2, G2, t), val.answer)]))
print(f"val F1 = {np.mean([f1(p, g) for p, g in zip(du_doan(val, S2, P2, G2, tot), val.answer)]):.4f}, ngưỡng để trống {tot:.2f}")

St, Ft, Gt, _ = tao_mau(te, False)      # dùng luôn mô hình 80% (train lại trên toàn bộ train được thêm chút điểm nhưng gấp đôi thời gian)
pd.DataFrame({"id": te.id, "answer": du_doan(te, St, m.predict(dv.transform(Ft)), Gt, tot)}).to_csv("submission.csv", index=False)
print("đã ghi submission.csv")
