"""Dựng các bài theo dạng còn thiếu so với bộ đề đầu (gọi qua xay_de.py, ví dụ `python tools/xay_de.py cv6 ml1`):
cv6 KTVIC_Captioning (caption ảnh tiếng Việt), cv7 CIFAR_Adversarial (tấn công đối kháng), au1 SpeechCommands_KWS (âm thanh),
ml1 Robot_PathPlanning (lập kế hoạch đường đi), ml2 HomeCredit_Tabular (dữ liệu bảng nhiều file), nlp8 ViQuAD_ExtractiveQA (hỏi đáp trích xuất).
Mỗi bài: test public để thẳng trong gói, test private nằm trong private_test.zip có mật khẩu (host/mat_khau_private_test.txt)."""
import base64
import hashlib
import heapq
import io
import json
import os
from collections import deque

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from chung import RAW, ghi_goi, ma_id, thu_muc, zip_mat_khau

SEED = int(os.environ["SELFHOST_SEED"])


def doc_pq(mau, cot=None):
    return pd.concat([pq.read_table(f, columns=cot).to_pandas() for f in sorted(RAW.glob(mau))], ignore_index=True)


def chia_khoi(nhom, rng, ti_le=0.5):
    """Public/Private theo khối (cả nhóm cùng một phía): nhóm xáo ngẫu nhiên, cộng dồn tới khi đủ tỉ lệ public."""
    nhom = pd.Series(nhom).astype(str)
    dem = nhom.value_counts()
    thu_tu = dem.index[rng.permutation(len(dem))]
    pub, tong = set(), 0
    for g in thu_tu:
        if tong >= ti_le * len(nhom):
            break
        pub.add(g); tong += dem[g]
    return np.where(nhom.isin(pub), "Public", "Private")


def mat_khau(up, host, khoa, cac_file):
    """Ghi private_test.zip (ZipCrypto) và host/mat_khau_private_test.txt. cac_file: list (tên trong zip, bytes)."""
    mk = hashlib.sha256(f"{khoa}-{SEED}".encode()).hexdigest()[:10]
    zip_mat_khau(up / "private_test.zip", sorted(cac_file), mk)
    (host / "mat_khau_private_test.txt").write_text(mk + "\n", encoding="utf-8")
    return mk


def ghi_license(up, text):
    (up / "LICENSE-DATASET.txt").write_text(text.strip() + "\n", encoding="utf-8")


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ======================================================================== CV6 caption ảnh tiếng Việt (KTVIC)
def cv6():
    up, host = thu_muc("KTVIC_Captioning")
    rng = np.random.default_rng(SEED)
    tr, te = doc_pq("ktvic/data/train-*.parquet"), doc_pq("ktvic/data/test-*.parquet")
    (up / "train/images").mkdir(parents=True); (up / "test/images").mkdir(parents=True)
    tid = ma_id(len(tr), SEED, "im")
    cap = {}
    for i, (_, r) in zip(tid, tr.iterrows()):
        (up / "train/images" / f"{i}.jpg").write_bytes(r.image["bytes"])
        cap[i] = [str(c) for c in r.caption_vi]
    (up / "train_captions.json").write_text(json.dumps(dict(sorted(cap.items())), ensure_ascii=False, indent=0), encoding="utf-8")
    eid = ma_id(len(te), SEED + 1, "te")
    usage = np.where(rng.permutation(len(te)) < len(te) // 2, "Public", "Private")
    rieng = []
    for i, u, (_, r) in zip(eid, usage, te.iterrows()):
        if u == "Public":
            (up / "test/images" / f"{i}.jpg").write_bytes(r.image["bytes"])
        else:
            rieng.append((f"private_test/images/{i}.jpg", r.image["bytes"]))
    mat_khau(up, host, "ktvic", rieng)
    sol = pd.DataFrame({"id": eid, "captions": [json.dumps([str(c) for c in r], ensure_ascii=False) for r in te.caption_vi], "Usage": usage})
    sol = sol.sort_values("id").reset_index(drop=True)
    ghi_license(up, """
KTVIC: A Vietnamese Image Captioning Dataset on the Life Domain (Pham và cộng sự, 2024), bản Hugging Face ai-enthusiasm-community/KTVIC.
Ảnh lấy từ UIT-EVJVQA. Chỉ dùng cho học tập và nghiên cứu, ghi nguồn khi trích dẫn.""")
    ghi_goi(up, host, sol, pd.DataFrame({"id": sol.id, "caption": "có một người đang đứng ở trên đường"}), "cider_d")
    print(f"  train {len(tr)} ảnh x 5 câu, test public {(usage == 'Public').sum()} / private {(usage == 'Private').sum()}")


# ======================================================================== CV7 tấn công đối kháng (CIFAR-10)
LOP_CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MO_HINH_PY = '''"""TinyCNN: mô hình A của đề (và mô hình ẩn B cùng kiến trúc). Đầu vào: tensor (N, 3, 32, 32) giá trị 0-1."""
import torch
from torch import nn

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0, self.c1, self.c2 = nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        h = (x - self.mean) / self.std
        for c in (self.c0, self.c1, self.c2):
            h = torch.max_pool2d(torch.relu(c(h)), 2)
        return self.fc(h.flatten(1))


def tai_mo_hinh(duong_dan="model_A.pt"):
    m = TinyCNN()
    m.load_state_dict(torch.load(duong_dan, map_location="cpu"), strict=False)
    return m.eval()
'''


def _train_tinycnn(x, y, seed, epochs=30):
    """Train TinyCNN có BatchNorm rồi gộp BN vào conv, làm tròn trọng số về float16 (để metric numpy khớp từng bit)."""
    import torch
    from torch import nn
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.ModuleList([nn.Conv2d(a, b, 3, padding=1) for a, b in [(3, 16), (16, 32), (32, 64)]])
            self.bn = nn.ModuleList([nn.BatchNorm2d(b) for b in (16, 32, 64)])
            self.fc = nn.Linear(1024, 10)

        def forward(self, h):
            for c, b in zip(self.c, self.bn):
                h = torch.max_pool2d(torch.relu(b(c(h))), 2)
            return self.fc(nn.functional.dropout(h.flatten(1), 0.3, self.training))

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
    X = torch.tensor(x, device=dev).permute(0, 3, 1, 2).float().div(255)
    Y = torch.tensor(y, device=dev)
    net = Net().to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    bs = 256; steps = epochs * (len(X) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=steps)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(X), device=dev).split(bs)[:len(X) // bs]:
            xb = X[idx]
            if g.random() < 0.5:
                xb = xb.flip(3)
            dy, dx = g.integers(-4, 5, 2)
            xb = torch.roll(xb, (int(dy), int(dx)), (2, 3))
            loss = nn.functional.cross_entropy(net((xb - mean) / std), Y[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    sd = {}
    for i, (c, b) in enumerate(zip(net.c, net.bn)):
        s = b.weight / torch.sqrt(b.running_var + b.eps)
        sd[f"c{i}.weight"] = (c.weight * s.view(-1, 1, 1, 1)).detach()
        sd[f"c{i}.bias"] = ((c.bias - b.running_mean) * s + b.bias).detach()
    sd["fc.weight"], sd["fc.bias"] = net.fc.weight.detach(), net.fc.bias.detach()
    return {k: v.cpu().numpy().astype(np.float16).astype(np.float32) for k, v in sd.items()}


def cv7():
    import torch
    from PIL import Image
    import metrics  # chung.py đã thêm selfhost_kaggle/ vào sys.path
    up, host = thu_muc("CIFAR_Adversarial")
    rng = np.random.default_rng(SEED)
    anh = lambda df: np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df.img])
    tr, te = doc_pq("cifar10/plain_text/train-*.parquet"), doc_pq("cifar10/plain_text/test-*.parquet")
    xtr, ytr, xte, yte = anh(tr), tr.label.values.astype(np.int64), anh(te), te.label.values.astype(np.int64)
    w = {"A": _train_tinycnn(xtr, ytr, SEED), "B": _train_tinycnn(xtr, ytr, SEED + 7)}
    pa, pb = metrics._cnn_np(xte, w["A"]), metrics._cnn_np(xte, w["B"])
    print(f"  độ chính xác test: A {np.mean(pa == yte):.3f}, B {np.mean(pb == yte):.3f}")
    chon = []  # 100 ảnh mỗi lớp mà cả A và B đều đoán đúng
    for c in range(10):
        ung = np.where((yte == c) & (pa == c) & (pb == c))[0]
        chon += list(rng.choice(ung, 100, replace=False))
    chon = np.array(chon)[rng.permutation(1000)]
    tid = ma_id(1000, SEED, "im")
    usage = np.where(np.arange(1000) < 500, "Public", "Private")
    (up / "test/images").mkdir(parents=True)
    rieng = []
    for i, u, k in zip(tid, usage, chon):
        b = io.BytesIO(); Image.fromarray(xte[k]).save(b, "PNG")
        if u == "Public":
            (up / "test/images" / f"{i}.png").write_bytes(b.getvalue())
        else:
            rieng.append((f"private_test/images/{i}.png", b.getvalue()))
    pd.DataFrame({"id": [i for i, u in zip(tid, usage) if u == "Public"], "label": [int(yte[k]) for k, u in zip(chon, usage) if u == "Public"]}).to_csv(up / "test_labels.csv", index=False)
    rieng_nhan = pd.DataFrame({"id": [i for i, u in zip(tid, usage) if u == "Private"], "label": [int(yte[k]) for k, u in zip(chon, usage) if u == "Private"]})
    mat_khau(up, host, "cifar-adv", rieng + [("private_test/test_labels.csv", csv_bytes(rieng_nhan))])
    np.savez_compressed(up / "cifar_train.npz", x=xtr, y=ytr)
    (up / "mo_hinh.py").write_text(MO_HINH_PY, encoding="utf-8")
    torch.save({k: torch.tensor(v) for k, v in w["A"].items()}, up / "model_A.pt")
    b64 = lambda a: base64.b64encode(a.astype(np.uint8).tobytes()).decode()
    sol = pd.DataFrame({"id": tid, "label": yte[chon], "orig": [b64(xte[k]) for k in chon], "Usage": usage}).sort_values("id")
    ghi_license(up, """
CIFAR-10 (Krizhevsky, 2009), bản Hugging Face uoft-cs/cifar10. Dùng cho học tập và nghiên cứu, ghi nguồn khi trích dẫn.
model_A.pt do người dựng đề tự train trên CIFAR-10 train.""")
    ghi_goi(up, host, sol, pd.DataFrame({"id": sol.id, "image": sol.orig}), "adv_success")
    ts = {ten: {k: [base64.b64encode(v.astype(np.float16).tobytes()).decode(), list(v.shape)] for k, v in m.items()} for ten, m in w.items()}
    p = host / "metric.py"
    p.write_text(p.read_text(encoding="utf-8").replace("_TRONG_SO = {}  # THAY_TRONG_SO", "_TRONG_SO = " + json.dumps(ts)), encoding="utf-8")
    print(f"  1000 ảnh test (500 public / 500 private), metric.py {p.stat().st_size / 1e3:.0f} KB")


# ======================================================================== AU1 nhận dạng từ khoá (Speech Commands)
LENH = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _doc_wav(b):
    from scipy.io import wavfile
    sr, a = wavfile.read(io.BytesIO(b))
    a = a.astype(np.float32) / 32768
    return np.pad(a, (0, max(0, 16000 - len(a))))[:16000]


def _nhieu(a, rng, kho_nhieu):
    """Trộn nhiễu (trắng, hồng, nâu, tiếng nói khác) với SNR 0-20 dB, dịch thời gian, đổi âm lượng."""
    k = rng.choice(["trang", "hong", "nau", "noi"])
    n = rng.standard_normal(16000).astype(np.float32)
    if k in ("hong", "nau"):
        f = np.fft.rfftfreq(16000); f[0] = f[1]
        n = np.fft.irfft(np.fft.rfft(n) / (np.sqrt(f) if k == "hong" else f), 16000).astype(np.float32)
    elif k == "noi":
        n = sum(kho_nhieu[j] for j in rng.choice(len(kho_nhieu), 3, replace=False))
    snr = rng.uniform(0, 20)
    p_a, p_n = np.mean(a ** 2) + 1e-8, np.mean(n ** 2) + 1e-8
    a = np.roll(a, int(rng.integers(-2000, 2001))) + n * np.sqrt(p_a / p_n / 10 ** (snr / 10))
    return a * rng.uniform(0.3, 1.0) / max(1e-3, np.abs(a).max())


def au1():
    from scipy.io import wavfile
    up, host = thu_muc("SpeechCommands_KWS")
    rng = np.random.default_rng(SEED)
    d = doc_pq("speech/data/*.parquet")
    d = d[d.label.astype(int) < 35].reset_index(drop=True)
    d["nhan"] = [LENH[int(l)] if int(l) < 10 else "unknown" for l in d.label]
    loa = d.speaker_id.unique()
    loa_test = set(rng.choice(loa, int(0.3 * len(loa)), replace=False))
    la_test = d.speaker_id.isin(loa_test).values
    tr, te = d[~la_test].reset_index(drop=True), d[la_test].reset_index(drop=True)
    (up / "train").mkdir(); (up / "test").mkdir()
    tid = ma_id(len(tr), SEED, "tr")
    for i, a in zip(tid, tr.audio):
        (up / "train" / f"{i}.wav").write_bytes(a["bytes"])
    ma_loa = {s: f"spk{j:03d}" for j, s in enumerate(rng.permutation(loa))}
    pd.DataFrame({"id": tid, "label": tr.nhan, "speaker": tr.speaker_id.map(ma_loa)}).sort_values("id").to_csv(up / "train.csv", index=False)
    kho = [_doc_wav(a["bytes"]) for a in tr.audio.sample(300, random_state=SEED)]
    eid = ma_id(len(te), SEED + 1, "te")
    usage = chia_khoi(te.speaker_id, rng)
    rieng = []
    for i, u, a in zip(eid, usage, te.audio):
        x = _doc_wav(a["bytes"])
        if rng.random() < 0.7:
            x = _nhieu(x, rng, kho)
        b = io.BytesIO(); wavfile.write(b, 16000, (np.clip(x, -1, 1) * 32767).astype(np.int16))
        if u == "Public":
            (up / "test" / f"{i}.wav").write_bytes(b.getvalue())
        else:
            rieng.append((f"private_test/{i}.wav", b.getvalue()))
    mat_khau(up, host, "speech", rieng)
    sol = pd.DataFrame({"id": eid, "label": te.nhan, "Usage": usage}).sort_values("id")
    ghi_license(up, """
Speech Commands v0.02 (Warden, 2018), bản Hugging Face mteb/speech-commands-mini. Giấy phép CC BY 4.0: ghi nguồn khi dùng.
Âm thanh test đã được trộn nhiễu tổng hợp.""")
    ghi_goi(up, host, sol, pd.DataFrame({"id": sol.id, "label": "unknown"}), "macro_f1")
    print(f"  train {len(tr)} ({len(loa) - len(loa_test)} người nói), test {len(te)} ({len(loa_test)} người nói khác), "
          f"{tr.nhan.value_counts().to_dict()}")


# ======================================================================== ML1 lập kế hoạch đường đi cho robot
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _ban_do(rng):
    H, W = (int(v) for v in rng.integers(20, 49, 2))
    z = rng.random((H, W))
    for _ in range(4):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    luoi = (1 + np.floor(z ** 1.5 * 8.999)).astype(int).astype(str).astype(object)
    luoi[rng.random((H, W)) < 0.08] = "#"
    for _ in range(H * W // 60):  # các đoạn tường dài
        y, x, L = int(rng.integers(0, H)), int(rng.integers(0, W)), int(rng.integers(4, 14))
        if rng.random() < 0.5:
            luoi[y, x:x + L] = "#"
        else:
            luoi[y:y + L, x] = "#"
    tu_do = [(y, x) for y in range(H) for x in range(W) if luoi[y, x] != "#"]
    tp, nhan = {}, 0
    for o in tu_do:  # thành phần liên thông lớn nhất
        if o in tp:
            continue
        q = deque([o]); tp[o] = nhan
        while q:
            y, x = q.popleft()
            for dy, dx in MOVES.values():
                v = (y + dy, x + dx)
                if 0 <= v[0] < H and 0 <= v[1] < W and luoi[v] != "#" and v not in tp:
                    tp[v] = nhan; q.append(v)
        nhan += 1
    lon = pd.Series(tp).value_counts().index[0]
    ung = [o for o in tu_do if tp[o] == lon]
    K = int(rng.integers(3, 11))
    diem = [ung[i] for i in rng.choice(len(ung), K + 2, replace=False)]
    for o, c in zip(diem, ["S", "G"] + [chr(65 + i) for i in range(K)]):
        luoi[o] = c
    return ["".join(r) for r in luoi]


def _dijkstra(luoi, nguon):
    H, W = len(luoi), len(luoi[0])
    dist, cha = {nguon: 0}, {}
    pq_ = [(0, nguon)]
    while pq_:
        d, u = heapq.heappop(pq_)
        if d > dist[u]:
            continue
        for c, (dy, dx) in MOVES.items():
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < H and 0 <= v[1] < W and luoi[v[0]][v[1]] != "#":
                o = luoi[v[0]][v[1]]
                nd = d + (int(o) if o.isdigit() else 1)
                if nd < dist.get(v, 1 << 30):
                    dist[v], cha[v] = nd, (u, c); heapq.heappush(pq_, (nd, v))
    return dist, cha


def giai_toi_uu(luoi):
    """Chi phí nhỏ nhất và chuỗi bước tối ưu: Dijkstra giữa các điểm then chốt + quy hoạch động trên tập điểm mốc."""
    vt = {c: (y, x) for y, r in enumerate(luoi) for x, c in enumerate(r) if c in "SG" or "A" <= c <= "L"}
    moc = sorted(c for c in vt if c not in "SG")
    K = len(moc)
    dj = {c: _dijkstra(luoi, vt[c]) for c in vt}
    d = lambda a, b: dj[a][0][vt[b]]
    INF = 1 << 40
    dp = {(1 << i, i): (d("S", moc[i]), None) for i in range(K)}
    for mask in range(1, 1 << K):
        for i in range(K):
            if (mask, i) not in dp:
                continue
            for j in range(K):
                if not mask >> j & 1:
                    nm, nv = mask | 1 << j, dp[(mask, i)][0] + d(moc[i], moc[j])
                    if nv < dp.get((nm, j), (INF,))[0]:
                        dp[(nm, j)] = (nv, i)
    full = (1 << K) - 1
    j = min(range(K), key=lambda i: dp[(full, i)][0] + d(moc[i], "G"))
    tot = dp[(full, j)][0] + d(moc[j], "G")
    thu_tu, mask = [], full
    while j is not None:
        thu_tu.append(moc[j]); pj = dp[(mask, j)][1]; mask ^= 1 << j; j = pj
    chuoi = ["S"] + thu_tu[::-1] + ["G"]
    buoc = ""
    for a, b in zip(chuoi, chuoi[1:]):
        cha, v, s = dj[a][1], vt[b], ""
        while v != vt[a]:
            v, c = cha[v]; s = c + s
        buoc += s
    return tot, buoc


def ml1():
    up, host = thu_muc("Robot_PathPlanning")
    rng = np.random.default_rng(SEED)
    n_tr, n_te = 300, 1000
    maps = [_ban_do(rng) for _ in range(n_tr + n_te)]
    sol_ = [giai_toi_uu(m) for m in maps]
    g = lambda m: "/".join(m)
    tid = ma_id(n_tr, SEED, "map")
    pd.DataFrame({"id": tid, "grid": [g(m) for m in maps[:n_tr]], "optimal_cost": [s[0] for s in sol_[:n_tr]],
                  "actions": [s[1] for s in sol_[:n_tr]]}).to_csv(up / "train.csv", index=False)
    eid = ma_id(n_te, SEED + 1, "te")
    usage = np.where(rng.permutation(n_te) < n_te // 2, "Public", "Private")
    te = pd.DataFrame({"id": eid, "grid": [g(m) for m in maps[n_tr:]]})
    te[usage == "Public"].sort_values("id").to_csv(up / "test.csv", index=False)
    mat_khau(up, host, "robot", [("private_test.csv", csv_bytes(te[usage == "Private"].sort_values("id")))])
    sol = te.assign(optimal_cost=[s[0] for s in sol_[n_tr:]], Usage=usage).sort_values("id")
    ghi_license(up, "Bản đồ do người dựng đề sinh ngẫu nhiên, dùng tự do.")
    ghi_goi(up, host, sol, pd.DataFrame({"id": sol.id, "actions": ""}), "path_score")
    print(f"  train {n_tr} bản đồ (kèm lời giải tối ưu), test {n_te}; trung bình {np.mean([len(s[1]) for s in sol_]):.0f} bước")


# ======================================================================== ML2 dữ liệu bảng nhiều file (Home Credit)
def ml2():
    up, host = thu_muc("HomeCredit_Tabular")
    rng = np.random.default_rng(SEED)
    a = doc_pq("homecredit/application_train_dated/*.parquet")
    thang = a.application_date.astype(str).str[:7]
    a = a[thang >= "2019-01"].reset_index(drop=True); thang = thang[thang >= "2019-01"].reset_index(drop=True)
    moi = dict(zip(a.SK_ID_CURR, ma_id(len(a), SEED, "kh")))  # đổi mã khách hàng: không tra ngược được nhãn gốc
    a["SK_ID_CURR"] = a.SK_ID_CURR.map(moi)
    a = a.rename(columns={"SK_ID_CURR": "id", "TARGET": "label"})
    a["application_date"] = a.application_date.astype(str).str[:10]
    tr = a[thang <= "2020-06"]
    te = a[thang > "2020-06"]
    usage = np.where(thang[thang > "2020-06"] <= "2020-09", "Public", "Private")  # public: quý 3, private: quý 4
    tr.sample(frac=1, random_state=SEED).to_csv(up / "application_train.csv", index=False)
    x = te.drop(columns="label")
    x[usage == "Public"].sort_values("id").to_csv(up / "application_test.csv", index=False)
    mat_khau(up, host, "homecredit", [("application_private_test.csv", csv_bytes(x[usage == "Private"].sort_values("id")))])
    for ten in ["bureau", "previous_application"]:
        b = doc_pq(f"homecredit/{ten}/*.parquet")
        b = b[b.SK_ID_CURR.isin(moi)].copy()
        b["SK_ID_CURR"] = b.SK_ID_CURR.map(moi)
        b.rename(columns={"SK_ID_CURR": "id"}).to_csv(up / f"{ten}.csv", index=False)
        print(f"  {ten}: {len(b)} dòng")
    sol = pd.DataFrame({"id": te.id.values, "label": te.label.values, "Usage": usage}).sort_values("id")
    ghi_license(up, """
Home Credit Default Risk (Home Credit Group, cuộc thi Kaggle 2018), bản Hugging Face mohameddhameem/home-credit-default-risk
(có thêm cột application_date). Chỉ dùng cho học tập nội bộ, không phát tán lại.""")
    ghi_goi(up, host, sol, pd.DataFrame({"id": sol.id, "score": 0.5}), "auroc")
    print(f"  train {len(tr)} hồ sơ (01/2019-06/2020, {tr.label.mean():.3f} vỡ nợ), test public {(usage == 'Public').sum()} / private {(usage == 'Private').sum()}")


# ======================================================================== NLP8 hỏi đáp trích xuất (ViQuAD 2.0)
def nlp8():
    up, host = thu_muc("ViQuAD_ExtractiveQA")
    rng = np.random.default_rng(SEED)
    tr, te = doc_pq("viquad/data/train-*.parquet"), doc_pq("viquad/data/validation-*.parquet")
    khong = lambda d: d.is_impossible.astype(str).str.lower().eq("true").values

    def dap_an(d):
        return [[""] if k else list(dict.fromkeys(str(t) for t in r["text"])) for k, r in zip(khong(d), d.answers)]

    dev_t = set(rng.choice(tr.title.unique(), 12, replace=False))
    for ten, d, s in [("train", tr[~tr.title.isin(dev_t)], SEED), ("dev", tr[tr.title.isin(dev_t)], SEED + 2)]:
        d = d.reset_index(drop=True)
        pd.DataFrame({"id": ma_id(len(d), s, "q"), "title": d.title, "context": d.context, "question": d.question,
                      "answer": [x[0] for x in dap_an(d)],
                      "answer_start": [-1 if k else int(r["answer_start"][0]) for k, r in zip(khong(d), d.answers)]}
                     ).to_csv(up / f"{ten}.csv", index=False)
        print(f"  {ten}: {len(d)} câu hỏi, {d.title.nunique()} bài, {khong(d).mean():.2f} không có đáp án")
    eid = ma_id(len(te), SEED + 1, "q")
    usage = chia_khoi(te.title, rng)
    x = pd.DataFrame({"id": eid, "title": te.title, "context": te.context, "question": te.question})
    x[usage == "Public"].sort_values("id").to_csv(up / "test.csv", index=False)
    mat_khau(up, host, "viquad-qa", [("private_test.csv", csv_bytes(x[usage == "Private"].sort_values("id")))])
    sol = pd.DataFrame({"id": eid, "answers": [json.dumps(a, ensure_ascii=False) for a in dap_an(te)], "Usage": usage}).sort_values("id")
    ghi_license(up, """
UIT-ViQuAD 2.0 (Nguyen và cộng sự, VLSP 2021), bản Hugging Face taidng/UIT-ViQuAD2.0. Chỉ dùng cho học tập và nghiên cứu.""")
    ghi_goi(up, host, sol, pd.DataFrame({"id": sol.id, "answer": ""}), "qa_f1")
    print(f"  test {len(te)} câu hỏi ({te.title.nunique()} bài): public {(usage == 'Public').sum()} / private {(usage == 'Private').sum()}")
