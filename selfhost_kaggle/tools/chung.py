"""Hàm dùng chung khi dựng đề: chia public/private, ghi gói, sinh metric.py cho Kaggle, RLE, làm hỏng ảnh."""
import inspect
import io
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[1]  # thư mục selfhost_kaggle/
sys.path.insert(0, str(GOC))
import metrics  # noqa: E402  (selfhost_kaggle/metrics.py)

RAW = Path(os.environ.get("SELFHOST_RAW", GOC / "raw"))  # dữ liệu gốc tải bằng tai_du_lieu.py


def thu_muc(bai):
    d = GOC / bai
    for con in ["upload", "host"]:
        if (d / con).exists():
            shutil.rmtree(d / con)
        (d / con).mkdir(parents=True)
    return d / "upload", d / "host"


def usage(n, seed=42, ti_le_public=0.5, khoi=1):
    """Chia Public/Private. khoi > 1: chia theo khối liền nhau (dùng khi mẫu kề nhau giống nhau)."""
    rng = np.random.default_rng(seed)
    so_khoi = (n + khoi - 1) // khoi
    pub = rng.permutation(so_khoi) < round(so_khoi * ti_le_public)
    return np.where(np.repeat(pub, khoi)[:n], "Public", "Private")


def ghi_goi(upload, host, solution, sample, ten_metric, cot_id="id"):
    """solution: DataFrame có cột Usage. sample: DataFrame file nộp mẫu."""
    solution.to_csv(host / "solution.csv", index=False)
    sample.to_csv(upload / "sample_submission.csv", index=False)
    sinh_metric(host, ten_metric)
    print(f"  solution {len(solution)} dòng ({(solution.Usage == 'Public').sum()} public), metric {ten_metric}")


def sinh_metric(host, ten_metric):
    """Ghi host/metric.py: phần chung + hàm metric của bài + score() có docstring (Kaggle hiển thị docstring này)."""
    phan_chung = inspect.getsource(metrics).split("# ---------------------------------------------------------------- phân loại")[0]
    phan_chung = '"""Metric cho cuộc thi nội bộ OlpAI thử (dán toàn bộ file này vào Kaggle Metric Notebook)."""' + phan_chung.split('"""', 2)[2]
    phu = []
    if ten_metric == "mean_dice":
        phu.append(inspect.getsource(metrics.rle_decode))
    if ten_metric == "recall_at_1":
        phu.append(inspect.getsource(metrics._recall))
    if ten_metric in ("bleu4",):
        phu.append(inspect.getsource(metrics._tach))
    f = getattr(metrics, ten_metric)
    doc = inspect.cleandoc(f.__doc__ or ten_metric).replace('"""', "'''")
    doc = "\n".join("    " + d if d else "" for d in doc.splitlines())
    code = (phan_chung + "\n".join(phu) + "\n\n" + inspect.getsource(f)
            + f'\n\ndef score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:\n    """\n{doc}\n    """\n'
            + f"    return {ten_metric}(solution, submission, row_id_column_name)\n")
    (host / "metric.py").write_text(code, encoding="utf-8")


def rle_encode(mask):
    """mask (H, W) 0/1 -> 'start length ...', start từ 1, duyệt theo hàng."""
    p = np.concatenate([[0], mask.flatten().astype(np.uint8), [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return " ".join(map(str, r))


def lam_hong(img, rng, muc=1.0):
    """PIL RGB -> PIL RGB bị làm hỏng ngẫu nhiên: JPEG nặng, mờ, nhiễu, đổi sáng/tương phản, che một mảng."""
    from PIL import Image, ImageEnhance, ImageFilter
    kieu = rng.choice(["jpeg", "blur", "noise", "light", "occlude"], size=rng.integers(1, 3), replace=False)
    for k in kieu:
        if k == "jpeg":
            b = io.BytesIO(); img.save(b, "JPEG", quality=int(rng.integers(8, 25))); img = Image.open(b).convert("RGB")
        elif k == "blur":
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(1.0, 2.5) * muc)))
        elif k == "noise":
            a = np.asarray(img).astype(np.float32) + rng.normal(0, 18 * muc, (img.height, img.width, 3))
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        elif k == "light":
            img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.45, 1.6)))
            img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.5, 1.4)))
        else:
            a = np.asarray(img).copy(); h, w = a.shape[:2]
            ch, cw = int(h * rng.uniform(0.15, 0.3)), int(w * rng.uniform(0.15, 0.3))
            y, x = rng.integers(0, h - ch), rng.integers(0, w - cw)
            a[y:y + ch, x:x + cw] = rng.integers(0, 255, 3)
            img = Image.fromarray(a)
    return img


def ma_id(n, seed, tien_to=""):
    """Sinh n id ngẫu nhiên không trùng, không lộ thứ tự gốc."""
    rng = np.random.default_rng(seed)
    so = rng.choice(10 ** 6, size=n, replace=False)
    return [f"{tien_to}{s:06d}" for s in so]


# ---------------------------------------------------------------- dùng cho bài Nordland (đề OlpAI'26)
def hex_id(n, rng, tien_to, da_dung):
    """n id dạng '<tiền tố>_7d1a0c' (6 ký tự hex ngẫu nhiên), không trùng với các id trong tập da_dung."""
    out = []
    while len(out) < n:
        s = f"{tien_to}_{int(rng.integers(0, 16 ** 6)):06x}"
        if s not in da_dung:
            da_dung.add(s); out.append(s)
    return out


def lam_hong_vpr(img, rng):
    """Ảnh mùa hè cần định vị bị làm hỏng như đề mô tả: tối, nhoè, sương mù, nén mạnh, độ phân giải thấp, mưa, chói.
    Mỗi ảnh chịu 1-2 kiểu ngẫu nhiên."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    w, h = img.size
    for k in rng.choice(["toi", "nhoe", "suong", "nen", "phan_giai", "mua", "choi"], size=int(rng.integers(1, 3)), replace=False):
        if k == "toi":
            a = np.asarray(img).astype(np.float32) / 255
            a = np.power(a, float(rng.uniform(1.8, 3.0))) * float(rng.uniform(0.35, 0.7))
            img = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
        elif k == "nhoe":
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(2.0, 5.0))))
        elif k == "suong":
            a = np.asarray(img).astype(np.float32)
            t = float(rng.uniform(0.35, 0.65))
            img = Image.fromarray((a * (1 - t) + 215 * t).clip(0, 255).astype(np.uint8))
        elif k == "nen":
            b = io.BytesIO(); img.save(b, "JPEG", quality=int(rng.integers(4, 12))); img = Image.open(b).convert("RGB")
        elif k == "phan_giai":
            f = int(rng.integers(4, 9))
            img = img.resize((w // f, h // f), Image.BILINEAR).resize((w, h), Image.NEAREST)
        elif k == "mua":
            lop = Image.new("L", (w, h), 0); d = ImageDraw.Draw(lop)
            for _ in range(int(rng.integers(250, 600))):
                x, y = int(rng.integers(0, w)), int(rng.integers(0, h)); L = int(rng.integers(10, 30))
                d.line([(x, y), (x + L // 4, y + L)], fill=int(rng.integers(120, 220)), width=1)
            lop = lop.filter(ImageFilter.GaussianBlur(0.6))
            img = Image.composite(Image.new("RGB", (w, h), (225, 225, 230)), ImageEnhance.Brightness(img).enhance(0.85), lop)
        else:  # chói: một vùng sáng lan ra từ một điểm phía trên ảnh
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy, r = rng.uniform(0, w), rng.uniform(0, h * 0.5), rng.uniform(0.25, 0.5) * w
            m = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))[..., None] * float(rng.uniform(0.7, 1.0))
            a = np.asarray(img).astype(np.float32)
            img = Image.fromarray((a * (1 - m) + 255 * m).clip(0, 255).astype(np.uint8))
    return img


def zip_mat_khau(duong_dan, cac_file, mat_khau):
    """Ghi zip mã hoá ZipCrypto (kiểu mã hoá zip truyền thống, đọc được bằng zipfile của Python với pwd=...,
    hay `unzip -P`). cac_file: list (tên trong zip, bytes). Không nén thêm (ảnh jpg đã nén sẵn)."""
    import struct
    import zlib
    tbl = [0] * 256
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        tbl[i] = c

    def crc(c, b):
        return (c >> 8) ^ tbl[(c ^ b) & 0xFF]

    os_rand = np.random.default_rng()
    pw = mat_khau.encode()
    central, off, f = [], 0, open(duong_dan, "wb")
    for ten, data in cac_file:
        k0, k1, k2 = 0x12345678, 0x23456789, 0x34567890

        def cap_nhat(b):
            nonlocal k0, k1, k2
            k0 = crc(k0, b); k1 = (k1 + (k0 & 0xFF)) & 0xFFFFFFFF
            k1 = (k1 * 134775813 + 1) & 0xFFFFFFFF; k2 = crc(k2, k1 >> 24)

        for b in pw:
            cap_nhat(b)
        c32 = zlib.crc32(data) & 0xFFFFFFFF
        head = bytes(os_rand.integers(0, 256, 11, dtype=np.uint8)) + bytes([c32 >> 24])
        out = bytearray()
        for b in head + data:
            t = (k2 | 2) & 0xFFFF
            out.append(b ^ (((t * (t ^ 1)) >> 8) & 0xFF)); cap_nhat(b)
        tb = ten.encode()
        loc = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 1, 0, 0, 0, c32, len(out), len(data), len(tb), 0)
        f.write(loc + tb + out)
        central.append(struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 1, 0, 0, 0, c32, len(out), len(data), len(tb), 0, 0, 0, 0, 0, off) + tb)
        off += len(loc) + len(tb) + len(out)
    cd = b"".join(central)
    f.write(cd + struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(central), len(central), len(cd), off, 0))
    f.close()
