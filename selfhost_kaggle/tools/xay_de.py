"""Dựng gói đề cho từng bài.  SELFHOST_SEED=<số bí mật> python tools/xay_de.py cv0 cv1 ... nlp7   (không đối số = tất cả)
Mỗi bài: <Bài>/upload (đưa lên Kaggle làm Data), <Bài>/host (solution.csv, metric.py; KHÔNG đưa cho thí sinh).

Seed quyết định id ngẫu nhiên, cách chia Public/Private, nhãn nhiễu và ảnh bị làm hỏng, nên chính là "chìa khoá" của đáp án:
đặt qua biến môi trường SELFHOST_SEED, không ghi vào repo. Seed của buổi thi 26/11 không công bố."""
try:  # torch phải nạp trước pandas/pyarrow trên Windows, nếu không lỗi DLL (dùng cho cv7)
    import torch  # noqa: F401
except ImportError:
    pass
import html
import io
import os
import re
import sys
import tarfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from chung import GOC, RAW, ghi_goi, hex_id, lam_hong, lam_hong_vpr, ma_id, rle_encode, thu_muc, usage, zip_mat_khau

if "SELFHOST_SEED" not in os.environ:
    sys.exit("Đặt biến môi trường SELFHOST_SEED (một số nguyên bí mật) trước khi dựng đề.")
SEED = int(os.environ["SELFHOST_SEED"])


def doc_pq(mau):
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted(RAW.glob(mau))], ignore_index=True)


def anh(b):
    return Image.open(io.BytesIO(b["bytes"] if isinstance(b, dict) else b))


def thu_nho(im, canh_dai):
    w, h = im.size
    s = canh_dai / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR) if s < 1 else im


# ======================================================================== CV0 Nordland
# Dựng theo đề chính thức tác vụ 2 vòng sơ loại OlpAI'26 "Định vị tàu hoả bốn mùa".
DOAN = {  # đoạn tuyến (theo thứ tự trong danh sách 27.592 khung đã lọc), các đoạn cách nhau 400 khung
    "train": (0, 14000), "validation": (14400, 17400), "public": (17800, 22300), "private": (22700, 27300)}


def cv0():
    import hashlib
    up, host = thu_muc("Nordland_PlaceRecognition")
    nord = Path(os.environ.get("SELFHOST_NORDLAND", RAW / "nordland"))
    anh_dir = nord / "anh640"  # tạo bằng giai_nen_nordland.py (jpg 640x360)
    ten = open(nord / "dataset_imageNames/nordland_imageNames.txt").read().split()
    pos = [int(t[7:12]) for t in ten]  # images-08377.png -> 8377: số thứ tự khung của bản quay
    rng = np.random.default_rng(SEED)
    dung = set()

    def chep(src, dst, hong):
        if hong:
            lam_hong_vpr(Image.open(src).convert("RGB"), rng).save(dst, quality=90)
        else:
            dst.write_bytes(src.read_bytes())

    sol_rows, private_files = [], []
    for doan, (a, b) in DOAN.items():
        idx = list(range(a, b))
        he = idx[5::10]  # ảnh mùa hè: 1 ảnh mỗi 10 khung; ảnh mùa đông: mọi khung
        hong = doan != "train"  # ảnh hè của validation và test bị làm hỏng, train giữ nguyên chất lượng
        if doan in ("train", "validation"):
            sid, wid = hex_id(len(he), rng, "s", dung), hex_id(len(idx), rng, "w", dung)
            (up / doan / "summer").mkdir(parents=True); (up / doan / "winter").mkdir(parents=True)
            for i, k in zip(he, sid):
                chep(anh_dir / "summer" / ten[i].replace(".png", ".jpg"), up / doan / "summer" / f"{k}.jpg", hong)
            for i, k in zip(idx, wid):
                chep(anh_dir / "winter" / ten[i].replace(".png", ".jpg"), up / doan / "winter" / f"{k}.jpg", False)
            pd.DataFrame({"summer_id": sid, "position": [pos[i] for i in he]}).sample(frac=1, random_state=SEED).to_csv(up / f"{doan}_summer.csv", index=False)
            pd.DataFrame({"winter_id": wid, "position": [pos[i] for i in idx]}).sample(frac=1, random_state=SEED).to_csv(up / f"{doan}_winter.csv", index=False)
            print(f"  {doan}: {len(he)} ảnh hè, {len(idx)} ảnh đông")
            continue
        qid, wid = hex_id(len(he), rng, "q", dung), hex_id(len(idx), rng, "w", dung)
        (up / "test/database").mkdir(parents=True, exist_ok=True); (up / "test/queries").mkdir(parents=True, exist_ok=True)
        for i, k in zip(idx, wid):
            chep(anh_dir / "winter" / ten[i].replace(".png", ".jpg"), up / "test/database" / f"{k}.jpg", False)
        wpos = np.array([pos[i] for i in idx])
        for i, k in zip(he, qid):
            src = anh_dir / "summer" / ten[i].replace(".png", ".jpg")
            if doan == "public":
                chep(src, up / "test/queries" / f"{k}.jpg", True)
            else:
                b_ = io.BytesIO(); lam_hong_vpr(Image.open(src).convert("RGB"), rng).save(b_, "JPEG", quality=90)
                private_files.append((f"private_test/queries/{k}.jpg", b_.getvalue()))
            sol_rows.append((k, " ".join(np.array(wid)[np.abs(wpos - pos[i]) <= 10]), "Public" if doan == "public" else "Private"))
        print(f"  {doan}: {len(he)} ảnh hè cần định vị, {len(idx)} ảnh đông vào test/database")
    mk = hashlib.sha256(f"nordland-{SEED}".encode()).hexdigest()[:10]
    private_files.sort()
    zip_mat_khau(up / "private_test.zip", private_files, mk)
    (host / "mat_khau_private_test.txt").write_text(mk + "\n", encoding="utf-8")
    sol = pd.DataFrame(sol_rows, columns=["query_id", "positives", "Usage"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    db = sorted(p.stem for p in (up / "test/database").glob("*.jpg"))
    sample = pd.DataFrame({"query_id": sol.query_id, "winter_id": rng.choice(db, len(sol))})
    (up / "LICENSE-DATASET.txt").write_text(
        "Nordland dataset (NRK, Sünderhauf và cộng sự 2013), bản trên Hugging Face Somayeh-h/Nordland.\n"
        "Giấy phép CC BY-NC-SA 4.0: chỉ dùng phi thương mại, ghi nguồn, chia sẻ cùng giấy phép.\n", encoding="utf-8")
    ghi_goi(up, host, sol, sample, "recall_at_1_vpr")
    print(f"  test/database: {len(db)} ảnh; mật khẩu private_test.zip ghi ở host/mat_khau_private_test.txt")


# ======================================================================== CV1 máy bay
def cv1():
    up, host = thu_muc("FGVC_Aircraft_Noisy")
    names = re.findall(r"'(\d+)': (.+)", (RAW / "aircraft_train/README.md").read_text(encoding="utf-8").split("name: family")[1].split("- name:")[0])
    ho = {int(k): v.strip() for k, v in names}
    tr, te = doc_pq("aircraft_train/data/*.parquet"), doc_pq("aircraft_test/data/*.parquet")
    rng = np.random.default_rng(SEED)

    def luu(df, thu, ids, hong):
        (up / thu).mkdir()
        for b, i, h in zip(df.image, ids, hong):
            im = anh(b).convert("RGB")
            im = im.crop((0, 0, im.width, im.height - 20))  # bỏ dải chữ bản quyền ở đáy ảnh
            im = thu_nho(im, 400)
            if h:
                im = lam_hong(im, rng)
            im.save(up / thu / f"{i}.jpg", quality=90)

    tid = ma_id(len(tr), SEED, "tr"); sid = ma_id(len(te), SEED + 1, "te")
    luu(tr, "train", tid, [False] * len(tr))
    hong = rng.random(len(te)) < 0.6
    luu(te, "test", sid, hong)
    # nhãn nhiễu: 15% ảnh train bị gán nhầm sang họ khác cùng hãng sản xuất (nếu có), không thì họ ngẫu nhiên
    y = tr.family.values.copy(); hang = tr.manufacturer.values
    cung_hang = {m: sorted(set(y[hang == m])) for m in set(hang)}
    nhieu = rng.random(len(y)) < 0.15
    for k in np.where(nhieu)[0]:
        lc = [c for c in cung_hang[hang[k]] if c != y[k]] or [c for c in ho if c != y[k]]
        y[k] = rng.choice(lc)
    pd.DataFrame({"id": tid, "label": [ho[c] for c in y]}).to_csv(up / "train.csv", index=False)
    pd.DataFrame({"id": sorted(sid)}).to_csv(up / "test.csv", index=False)
    pd.DataFrame({"label": sorted(ho.values())}).to_csv(up / "classes.csv", index=False)
    sol = pd.DataFrame({"id": sid, "label": [ho[c] for c in te.family], "Usage": usage(len(te), SEED)})
    sample = pd.DataFrame({"id": sid, "label": "Boeing 737"})
    ghi_goi(up, host, sol, sample, "macro_f1")
    print(f"  train {len(tr)} ({nhieu.mean():.0%} nhãn nhiễu), test {len(te)} ({hong.mean():.0%} bị làm hỏng), {len(ho)} lớp")


# ======================================================================== CV2 polyp
def cv2():
    up, host = thu_muc("Kvasir_PolypSegmentation")
    df = doc_pq("kvasir/data/*.parquet").sample(frac=1, random_state=SEED).reset_index(drop=True)
    S = 256
    ids = ma_id(len(df), SEED, "im")
    n_tr = 800
    for d in ["train/images", "train/masks", "test/images"]:
        (up / d).mkdir(parents=True)
    rles = []
    for k, r in df.iterrows():
        im = anh(r.image).convert("RGB").resize((S, S), Image.BILINEAR)
        m = (np.asarray(anh(r.annotation).convert("L").resize((S, S), Image.NEAREST)) > 127).astype(np.uint8)
        if k < n_tr:
            im.save(up / f"train/images/{ids[k]}.jpg", quality=95)
            Image.fromarray(m * 255).save(up / f"train/masks/{ids[k]}.png")
        else:
            im.save(up / f"test/images/{ids[k]}.jpg", quality=95)
            rles.append(rle_encode(m))
    te = ids[n_tr:]
    pd.DataFrame({"id": sorted(ids[:n_tr])}).to_csv(up / "train.csv", index=False)
    pd.DataFrame({"id": sorted(te)}).to_csv(up / "test.csv", index=False)
    sol = pd.DataFrame({"id": te, "rle": rles, "height": S, "width": S, "Usage": usage(len(te), SEED)})
    c = np.zeros((S, S), np.uint8); c[64:192, 64:192] = 1
    sample = pd.DataFrame({"id": te, "rle": rle_encode(c)})
    ghi_goi(up, host, sol, sample, "mean_dice")


# ======================================================================== CV3 bất thường
def cv3():
    import json
    up, host = thu_muc("MVTec_AnomalyDetection")
    s = json.load(open(RAW / "mvtec/samples.json"))["samples"]
    CAT = ["bottle", "carpet", "hazelnut", "metal_nut", "screw"]
    s = [x for x in s if x["category"]["label"] in CAT]
    tr = [x for x in s if x["split"] == "train"]; te = [x for x in s if x["split"] == "test"]
    tid, sid = ma_id(len(tr), SEED, "tr"), ma_id(len(te), SEED + 1, "te")
    for d in ["train", "test"]:
        (up / d).mkdir()
    for xs, ids, d in [(tr, tid, "train"), (te, sid, "test")]:
        for x, i in zip(xs, ids):
            Image.open(RAW / "mvtec" / x["filepath"]).convert("RGB").resize((256, 256), Image.BILINEAR).save(up / f"{d}/{i}.png")
    pd.DataFrame({"id": tid, "category": [x["category"]["label"] for x in tr]}).to_csv(up / "train.csv", index=False)
    cat_te = [x["category"]["label"] for x in te]
    pd.DataFrame({"id": sid, "category": cat_te}).sort_values("id").to_csv(up / "test.csv", index=False)
    lab = [int(x["defect"]["label"] != "good") for x in te]
    # chia public/private phân tầng theo (loại, nhãn)
    u = np.empty(len(te), dtype=object)
    for key in set(zip(cat_te, lab)):
        idx = [k for k, z in enumerate(zip(cat_te, lab)) if z == key]
        u[idx] = usage(len(idx), SEED + len(idx))
    sol = pd.DataFrame({"id": sid, "label": lab, "Usage": u})
    sample = pd.DataFrame({"id": sid, "score": 0.5})
    ghi_goi(up, host, sol, sample, "auroc")
    print(f"  train {len(tr)} ảnh bình thường, test {len(te)} ({np.mean(lab):.0%} lỗi)")


# ======================================================================== CV4 video
def cv4():
    up, host = thu_muc("UCF101_ActionRecognition")
    (up / "train").mkdir(); (up / "test").mkdir()
    rows_tr, rows_te = [], []
    with tarfile.open(RAW / "ucf/UCF101_subset.tar.gz") as t:
        mem = [m for m in t.getmembers() if m.name.endswith(".avi")]
        sid = iter(ma_id(len(mem), SEED, "v"))
        for m in sorted(mem, key=lambda m: m.name):
            f = m.name.split("/")[-1]; lop = m.name.split("/")[-2]
            g = int(re.search(r"_g(\d+)_c", f).group(1))
            data = t.extractfile(m).read()
            if g <= 17:
                (up / "train" / f).write_bytes(data); rows_tr.append((f[:-4], lop))
            else:
                i = next(sid); (up / "test" / f"{i}.avi").write_bytes(data); rows_te.append((i, lop))
    pd.DataFrame(rows_tr, columns=["id", "label"]).to_csv(up / "train.csv", index=False)
    te = pd.DataFrame(rows_te, columns=["id", "label"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    pd.DataFrame({"id": sorted(te.id)}).to_csv(up / "test.csv", index=False)
    sol = te.assign(Usage=usage(len(te), SEED))
    ghi_goi(up, host, sol, pd.DataFrame({"id": te.id, "label": rows_tr[0][1]}), "macro_f1")
    print(f"  train {len(rows_tr)} clip (nhóm g01-g17), test {len(te)} clip (nhóm g18-g25)")


# ======================================================================== CV5 đa nhãn
VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
       "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def _mau_voc(i):
    """Bảng màu chuẩn của mask VOC: lớp i -> (R, G, B)."""
    r = g = b = 0
    for j in range(8):
        r |= ((i >> 0) & 1) << (7 - j); g |= ((i >> 1) & 1) << (7 - j); b |= ((i >> 2) & 1) << (7 - j); i >>= 3
    return (r, g, b)


MAU_VOC = [_mau_voc(i) for i in range(21)]


def cv5():
    up, host = thu_muc("VOC_MultiLabel")
    out = {}
    for sp, mau in [("train", "voc/data/train-*.parquet"), ("test", "voc/data/val-*.parquet")]:
        df = doc_pq(mau)
        ids = ma_id(len(df), SEED + (sp == "test"), sp[:2])
        (up / sp).mkdir()
        labs = []
        for b, mb, i in zip(df.image, df["mask"], ids):
            thu_nho(anh(b).convert("RGB"), 384).save(up / sp / f"{i}.jpg", quality=92)
            m = np.asarray(anh(mb).convert("RGB")).reshape(-1, 3)
            mau = {tuple(x) for x in np.unique(m, axis=0)}
            labs.append(" ".join(VOC[c] for c in range(1, 21) if MAU_VOC[c] in mau))
        out[sp] = pd.DataFrame({"id": ids, "labels": labs})
    out["train"].to_csv(up / "train.csv", index=False)
    pd.DataFrame({"id": sorted(out["test"].id)}).to_csv(up / "test.csv", index=False)
    pd.DataFrame({"label": VOC[1:]}).to_csv(up / "classes.csv", index=False)
    sol = out["test"].assign(Usage=usage(len(out["test"]), SEED))
    ghi_goi(up, host, sol, pd.DataFrame({"id": sol.id, "labels": "person"}), "multilabel_macro_f1")
    print(f"  train {len(out['train'])}, test {len(sol)}, trung bình {out['train'].labels.str.split().str.len().mean():.2f} nhãn/ảnh")


# ======================================================================== NLP1 cảm xúc
def nlp1():
    up, host = thu_muc("VSFC_Sentiment")
    tr = pd.concat([pd.read_csv(RAW / f"vsfc/UIT-VSFC_{s}.csv") for s in ["train", "valid"]], ignore_index=True)
    te = pd.read_csv(RAW / "vsfc/UIT-VSFC_test.csv")
    tr = tr.dropna(subset=["text"]).reset_index(drop=True); te["text"] = te.text.fillna("")
    tr.insert(0, "id", ma_id(len(tr), SEED, "tr")); te.insert(0, "id", ma_id(len(te), SEED + 1, "te"))
    tr.to_csv(up / "train.csv", index=False)
    te[["id", "text"]].sort_values("id").to_csv(up / "test.csv", index=False)
    sol = te[["id", "label"]].assign(Usage=usage(len(te), SEED))
    ghi_goi(up, host, sol, pd.DataFrame({"id": te.id, "label": "positive"}), "macro_f1")
    print(f"  train {len(tr)} {tr.label.value_counts().to_dict()}, test {len(te)}")


# ======================================================================== NLP2 NER
def nlp2():
    up, host = thu_muc("PhoNER_COVID19")
    tr = pd.concat([doc_pq(f"phoner/syllable/{s}-*.parquet") for s in ["train", "validation"]], ignore_index=True)
    te = doc_pq("phoner/syllable/test-*.parquet")
    j = lambda xs: [" ".join(x) for x in xs]
    tr = pd.DataFrame({"id": ma_id(len(tr), SEED, "tr"), "words": j(tr.words), "tags": j(tr.tags)})
    te = pd.DataFrame({"id": ma_id(len(te), SEED + 1, "te"), "words": j(te.words), "tags": j(te.tags)})
    tr.to_csv(up / "train.csv", index=False)
    te[["id", "words"]].sort_values("id").to_csv(up / "test.csv", index=False)
    sol = te[["id", "tags"]].assign(Usage=usage(len(te), SEED))
    sample = pd.DataFrame({"id": te.id, "tags": [" ".join(["O"] * len(w.split())) for w in te.words]})
    ghi_goi(up, host, sol, sample, "entity_f1")
    print(f"  train {len(tr)} câu, test {len(te)} câu")


# ======================================================================== NLP3 thêm dấu
def bo_dau(s):
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def nlp3():
    up, host = thu_muc("Vietnamese_Diacritics")
    df = doc_pq("news/data/*.parquet")
    rng = np.random.default_rng(SEED)
    cau = []
    for s in pd.concat([df.title, df.description]).dropna().sample(frac=1, random_state=SEED).head(400000):
        s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()
        if 6 <= len(s.split()) <= 40 and bo_dau(s) != s:
            cau.append(s)
    cau = list(dict.fromkeys(cau))
    tr, te = cau[:100000], cau[100000:105000]
    pd.DataFrame({"id": ma_id(len(tr), SEED, "tr"), "text_khong_dau": [bo_dau(s) for s in tr], "text": tr}).to_csv(up / "train.csv", index=False)
    tid = ma_id(len(te), SEED + 1, "te")
    pd.DataFrame({"id": tid, "text_khong_dau": [bo_dau(s) for s in te]}).sort_values("id").to_csv(up / "test.csv", index=False)
    sol = pd.DataFrame({"id": tid, "text": te, "Usage": usage(len(te), SEED)})
    ghi_goi(up, host, sol, pd.DataFrame({"id": tid, "text": [bo_dau(s) for s in te]}), "word_accuracy")
    print(f"  train {len(tr)} câu, test {len(te)} câu")


# ======================================================================== NLP4 dịch
def nlp4():
    up, host = thu_muc("IWSLT_EnVi_Translation")
    sach = lambda s: re.sub(r"\s+", " ", html.unescape(unicodedata.normalize("NFC", str(s)))).strip()
    tr = doc_pq("iwslt/data/train-*.parquet"); te = doc_pq("iwslt/data/test-*.parquet")
    for d in (tr, te):
        d["en"] = d.en.map(sach); d["vi"] = d.vi.map(sach)
    tr = tr[(tr.en.str.len() > 0) & (tr.vi.str.len() > 0)].reset_index(drop=True)
    tr.insert(0, "id", ma_id(len(tr), SEED, "tr")); tr.to_csv(up / "train.csv", index=False)
    tid = ma_id(len(te), SEED + 1, "te")
    pd.DataFrame({"id": tid, "en": te.en}).sort_values("id").to_csv(up / "test.csv", index=False)
    sol = pd.DataFrame({"id": tid, "vi": te.vi, "Usage": usage(len(te), SEED)})
    ghi_goi(up, host, sol, pd.DataFrame({"id": tid, "vi": te.en}), "bleu4")
    print(f"  train {len(tr)} cặp, test {len(te)} câu")


# ======================================================================== NLP5 truy xuất
def nlp5():
    up, host = thu_muc("ViQuAD_PassageRetrieval")
    v = pd.concat([doc_pq(f"viquad/data/{s}-*.parquet").assign(split=s) for s in ["train", "validation", "test"]], ignore_index=True)
    v["context"] = v.context.map(lambda s: unicodedata.normalize("NFC", s))
    ctx = v.drop_duplicates("context")[["context", "title"]].reset_index(drop=True)
    ctx["passage_id"] = ma_id(len(ctx), SEED, "p")
    v = v.merge(ctx[["context", "passage_id"]], on="context")
    ctx.rename(columns={"context": "text"})[["passage_id", "title", "text"]].sort_values("passage_id").to_csv(up / "corpus.csv", index=False)
    tr = v[v.split == "train"]
    tr = pd.DataFrame({"id": ma_id(len(tr), SEED + 1, "tr"), "question": tr.question.values, "passage_id": tr.passage_id.values})
    tr.to_csv(up / "train.csv", index=False)
    te = v[v.split != "train"].sample(n=4000, random_state=SEED)
    tid = ma_id(len(te), SEED + 2, "q")
    pd.DataFrame({"id": tid, "question": te.question.values}).sort_values("id").to_csv(up / "test.csv", index=False)
    sol = pd.DataFrame({"id": tid, "passage_id": te.passage_id.values, "Usage": usage(len(te), SEED)})
    rng = np.random.default_rng(SEED)
    ghi_goi(up, host, sol, pd.DataFrame({"id": tid, "passage_ids": [" ".join(rng.choice(ctx.passage_id, 10, replace=False)) for _ in tid]}), "mrr_at_10")
    print(f"  corpus {len(ctx)} đoạn, train {len(tr)} câu hỏi, test {len(te)} câu hỏi")


# ======================================================================== NLP6, NLP7 DimABSA (đề NLP OlpAI'26)
# Đề thật: tiếng Việt, dữ liệu không công bố. Bản này dùng dữ liệu công khai SemEval-2026 Task 3 (DimABSA2026,
# Track A, eng-restaurant), đổi tên trường sang đúng định dạng đề (id, text, aspects / quadruplets, valence, arousal).
def _dimabsa_doc(ten):
    import json
    d = Path(os.environ.get("SELFHOST_DIMABSA", RAW / "dimabsa"))
    return [json.loads(l) for l in open(d / ten, encoding="utf-8") if l.strip()]


def _va(s):
    v, a = s.split("#")
    return float(v), float(a)


def _dimabsa_goi(bai, khoa, train, dev, test, cot_nop, mau_rong):
    """Ghi gói chung cho hai subtask. test: list (đầu vào cho thí sinh, đáp án). Nửa đầu sau khi xáo là public
    (test.jsonl), nửa sau là private (private_test.zip có mật khẩu, giải nén ra private_test.jsonl)."""
    import hashlib
    import json
    up, host = thu_muc(bai)
    rng = np.random.default_rng(SEED)
    ghi = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for ten, rows in [("train", train), ("dev", dev)]:
        (up / f"{ten}.jsonl").write_text(ghi(rows), encoding="utf-8")
    test = [test[i] for i in rng.permutation(len(test))]
    n_pub = len(test) // 2
    pub, prv = test[:n_pub], test[n_pub:]
    for j, (x, _) in enumerate(pub):
        x["id"] = f"PUB_{j + 1:04d}"
    for j, (x, _) in enumerate(prv):
        x["id"] = f"PRV_{j + 1:04d}"
    (up / "test.jsonl").write_text(ghi([x for x, _ in pub]), encoding="utf-8")
    mk = hashlib.sha256(f"{khoa}-{SEED}".encode()).hexdigest()[:10]
    zip_mat_khau(up / "private_test.zip", [("private_test.jsonl", ghi([x for x, _ in prv]).encode("utf-8"))], mk)
    (host / "mat_khau_private_test.txt").write_text(mk + "\n", encoding="utf-8")
    sol = pd.DataFrame([(x["id"], json.dumps(y, ensure_ascii=False), "Public" if x["id"].startswith("PUB") else "Private")
                        for x, y in pub + prv], columns=["id", cot_nop, "Usage"])
    sample = pd.DataFrame({"id": sol.id, cot_nop: [json.dumps(mau_rong(x), ensure_ascii=False) for x, _ in pub + prv]})
    (up / "LICENSE-DATASET.txt").write_text(
        "Dữ liệu DimABSA2026 (SemEval-2026 Task 3, Track A, eng-restaurant), github.com/DimABSA/DimABSA2026, commit bdc93be.\n"
        "Kho gốc không ghi giấy phép: chỉ dùng để luyện tập nội bộ, không phát tán lại, ghi nguồn khi trích dẫn.\n", encoding="utf-8")
    ghi_goi(up, host, sol, sample, "rmse_va" if cot_nop == "aspects" else "continuous_f1")
    print(f"  train {len(train)} câu, dev {len(dev)}, test public {len(pub)} / private {len(prv)}; mật khẩu ở host/mat_khau_private_test.txt")


def nlp6():
    """Subtask 1: cho câu và các khía cạnh, dự đoán (valence, arousal) cho từng khía cạnh."""
    def cap(rows, k_va="Aspect_VA"):
        out = []
        for j, r in enumerate(rows):
            asp = [{"aspect": q["Aspect"], "valence": _va(q["VA"])[0], "arousal": _va(q["VA"])[1]}
                   for q in r[k_va] if q["Aspect"] != "NULL"]
            if asp:
                out.append({"id": "", "text": r["Text"], "aspects": asp})
        return out

    train = cap(_dimabsa_doc("eng_restaurant_train_alltasks.jsonl"), "Quadruplet")
    for j, r in enumerate(train):
        r["id"] = f"TRAIN_{j + 1:04d}"
    dev = cap(_dimabsa_doc("eng_restaurant_dev_task1.jsonl"))
    for j, r in enumerate(dev):
        r["id"] = f"DEV_{j + 1:04d}"
    test = [({"id": "", "text": r["text"], "aspects": [{"term": a["aspect"]} for a in r["aspects"]]}, r["aspects"])
            for r in cap(_dimabsa_doc("eng_restaurant_test_task1.jsonl"))]
    mau = lambda x: [{"aspect": a["term"], "valence": 5.0, "arousal": 5.0} for a in x["aspects"]]
    _dimabsa_goi("DimABSA_VA", "dimabsa-va", train, dev, test, "aspects", mau)


def nlp7():
    """Subtask 2: chỉ cho câu, trích tập bộ tứ (aspect, category, opinion, valence, arousal)."""
    def cap(rows):
        return [{"id": "", "text": r["Text"], "quadruplets": [
            {"aspect": q["Aspect"], "category": q["Category"], "opinion": q["Opinion"],
             "valence": _va(q["VA"])[0], "arousal": _va(q["VA"])[1]} for q in r["Quadruplet"]]} for r in rows]

    train = cap(_dimabsa_doc("eng_restaurant_train_alltasks.jsonl"))
    for j, r in enumerate(train):
        r["id"] = f"TRAIN_{j + 1:04d}"
    dev = cap(_dimabsa_doc("eng_restaurant_dev_task3.jsonl"))
    for j, r in enumerate(dev):
        r["id"] = f"DEV_{j + 1:04d}"
    test = [({"id": "", "text": r["text"]}, r["quadruplets"]) for r in cap(_dimabsa_doc("eng_restaurant_test_task3.jsonl"))]
    _dimabsa_goi("DimABSA_Quad", "dimabsa-quad", train, dev, test, "quadruplets", lambda x: [])


from xay_de_them import au1, cv6, cv7, ml1, ml2, nlp8  # noqa: E402  (các dạng bài thêm sau)

if __name__ == "__main__":
    bai = sys.argv[1:] or ["cv0", "cv1", "cv2", "cv3", "cv4", "cv5", "nlp1", "nlp2", "nlp3", "nlp4", "nlp5", "nlp6", "nlp7", "nlp8", "cv6", "cv7", "au1", "ml1", "ml2"]
    for b in bai:
        print(b, flush=True)
        globals()[b]()
