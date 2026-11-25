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
