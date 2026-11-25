"""Bộ chấm dùng chung: kiểm tra quy tắc code, chạy bài làm, đọc file nộp."""
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class Bang:
    """Gom kết quả từng mục kiểm tra. Mục 'canh_bao' không làm trượt bài."""

    def __init__(self, ten):
        self.ten, self.rows, self.truot = ten, [], 0

    def ok(self, dk, msg, canh_bao=False):
        dk = bool(dk)
        tag = "ĐẠT " if dk else ("LƯU Ý" if canh_bao else "TRƯỢT")
        if not dk and not canh_bao:
            self.truot += 1
        self.rows.append(f"  [{tag}] {msg}")
        print(self.rows[-1], flush=True)
        return dk

    def ket_thuc(self):
        print(f"\n{self.ten}: " + ("ĐẠT tất cả" if self.truot == 0 else f"TRƯỢT {self.truot} mục"))
        sys.exit(0 if self.truot == 0 else 1)


def doi_so(mac_dinh="bai_lam.py"):
    """python check_xx.py [bai_lam.py] [--khong-chay]"""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    return Path(args[0] if args else mac_dinh).resolve(), "--khong-chay" not in sys.argv


def kiem_tra_code(B, path, torch=True, nfc=False, tu_khoa=()):
    """Quy tắc chung. tu_khoa: [(regex, mô tả)] chỉ cảnh báo nếu thiếu."""
    if not B.ok(path.exists(), f"có file bài làm {path.name}"):
        B.ket_thuc()
    src = path.read_text(encoding="utf-8")
    B.ok(re.search(r"seed\w*\(\s*(\w+\s*=\s*)?42\s*\)", src), "gọi seed_all(42)")
    if torch:
        B.ok(re.search(r"\.eval\(\)", src), "có model.eval() trước khi dự đoán")
        B.ok("inference_mode" in src, "dự đoán trong torch.inference_mode()")
        if "bfloat16" in src or "autocast" in src:
            B.ok("get_device_capability" in src, "bf16/autocast chỉ bật khi get_device_capability()[0] >= 8")
    if nfc:
        B.ok(re.search(r"normalize\(\s*['\"]NFC['\"]", src), "chuẩn hoá unicodedata.normalize('NFC', ...)")
    tree = ast.parse(src)
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
    sai = set()
    for lp in loops:
        for c in ast.walk(lp):
            if isinstance(c, ast.Call):
                ten = ast.unparse(c.func)
                if re.search(r"(^|\.)optim\.[A-Z]\w*$|lr_scheduler\.[A-Z]\w*$|^(SGD|Adam|AdamW|OneCycleLR|CosineAnnealingLR)$", ten):
                    sai.add(c.lineno)
    B.ok(not sai, "tạo optimizer/scheduler một lần, ngoài vòng lặp" + (f" (sai ở dòng {sorted(sai)})" if sai else ""))
    if torch:
        trong = set()
        for lp in loops:
            for inner in ast.walk(lp):
                if inner is not lp and isinstance(inner, ast.For):
                    for c in ast.walk(inner):
                        if isinstance(c, ast.Call) and re.search(r"(\.eval$|evaluate|validat)", ast.unparse(c.func)):
                            trong.add(c.lineno)
        B.ok(not trong, "validation nằm ngoài vòng batch" + (f" (xem dòng {sorted(trong)})" if trong else ""), canh_bao=True)
    rs = [i + 1 for i, l in enumerate(src.splitlines()) if re.search(r"\.(reshape|view)\(", l) and not l.lstrip().startswith("#")]
    if rs:
        B.ok(False, f"dòng {rs} dùng reshape/view: nếu là đổi thứ tự trục thì phải dùng permute/transpose", canh_bao=True)
    for rx, mo_ta in tu_khoa:
        B.ok(re.search(rx, src), mo_ta, canh_bao=True)
    return src


def chay(B, path, timeout_phut):
    """Chạy bài làm trong thư mục của bài, đo thời gian."""
    print(f"\n  ... chạy {path.name} (tối đa {timeout_phut} phút)", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        r = subprocess.run([sys.executable, str(path)], cwd=path.parent, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout_phut * 60, env=env)
    except subprocess.TimeoutExpired:
        B.ok(False, f"chạy xong trong {timeout_phut} phút")
        B.ket_thuc()
    phut = (time.time() - t0) / 60
    if r.returncode & 0xFFFFFFFF == 0xC0000409 and not r.stderr.strip():
        # Windows: một số tổ hợp torch (RNN) + numpy làm tiến trình crash lúc giải phóng DLL khi thoát, sau khi đã chạy xong.
        # Máy thi Linux không gặp. Chấm tiếp file nộp, chỉ cảnh báo.
        B.ok(False, f"chạy xong ({phut:.1f} phút) nhưng Windows báo lỗi 0xC0000409 lúc thoát (lỗi DLL của torch), vẫn chấm file nộp", canh_bao=True)
        return phut
    if r.returncode != 0:
        print("\n".join(r.stderr.strip().splitlines()[-15:]))
    B.ok(r.returncode == 0, f"chạy không lỗi ({phut:.1f} phút)")
    if r.returncode != 0:
        B.ket_thuc()
    return phut


def doc_csv(B, path, cot, ids, cot_id="id"):
    """Đọc file nộp: đúng tên cột, mỗi id đúng một dòng, không thiếu/thừa, không NaN. Trả DataFrame theo thứ tự ids."""
    if not B.ok(path.exists(), f"có file nộp {path.name}"):
        B.ket_thuc()
    df = pd.read_csv(path, dtype={cot_id: str}, keep_default_na=False, na_values=[""])
    B.ok(list(df.columns) == cot, f"{path.name}: tiêu đề đúng {cot} (đang là {list(df.columns)})")
    B.ok(df[cot_id].is_unique, f"{path.name}: mỗi id một dòng")
    thieu, thua = set(ids) - set(df[cot_id]), set(df[cot_id]) - set(ids)
    B.ok(not thieu and not thua, f"{path.name}: đủ {len(ids)} id, không thiếu/thừa (thiếu {len(thieu)}, thừa {len(thua)})")
    if thieu or not df[cot_id].is_unique:
        B.ket_thuc()
    df = df.drop_duplicates(cot_id).set_index(cot_id).loc[list(ids)]
    B.ok(not df.isna().any().any(), f"{path.name}: không có ô trống / NaN")
    return df


def doc_dap_an(thu_muc, ten):
    return pd.read_csv(Path(thu_muc) / ".dap_an" / ten, dtype={"id": str}, keep_default_na=False)
