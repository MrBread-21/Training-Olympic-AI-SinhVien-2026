"""Kiểm tra bộ bài: chép lời giải mẫu của mọi ngày ra file tạm, chấm, rồi xoá kết quả. Dùng sau khi sửa bộ chấm/dữ liệu.
  python chung/chay_tat_ca.py            (từ thư mục luyen_tap/CV)"""
import shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAI = {"d21_phan_loai_anh": ["giai.py"], "d22_augment_nhan_nhieu": ["giai.py"], "d23_dac_trung_dong_bang": ["giai.py"],
       "d24_truy_xuat": ["giai.py"], "d25_doi_khang": ["giai.py"], "d26_video": ["giai.py"],
       "d27_patchcore_unet": ["giai_a.py", "giai_b.py"], "d28_nlp": ["giai_a.py", "giai_a2.py", "giai_b.py"],
       "d29_seq2seq": ["giai.py"], "d30_du_lieu_bang": ["giai.py"],
       "d01_am_thanh": ["giai.py"], "d02_caption": ["giai_a.py", "giai_b.py"], "d03_hoi_dap": ["giai.py"],
       "d04_dimabsa": ["giai_a.py", "giai_b.py"], "d05_duong_di": ["giai_a.py", "giai_b.py"]}
RAC = ["pred_*.jsonl", "submission*.csv", "oof.csv", "*.npy", "*.npz", "best*.pt", "cam.npy", "adv", "masks_pred", "_mau_*.py"]
r = subprocess.run([sys.executable, "-X", "utf8", "kiem_tra.py", "--loi_giai"], cwd=ROOT / "d20_tinh_toan", capture_output=True, text=True, encoding="utf-8")
print("d20_tinh_toan:", r.stdout.strip().splitlines()[-1])
for d, files in BAI.items():
    tmp = [f"_mau_{f}" for f in files]
    for f, t in zip(files, tmp): shutil.copy(ROOT / d / "loi_giai" / f, ROOT / d / t)
    check = next((ROOT / d).glob("check_*.py"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", check.name, *tmp], cwd=ROOT / d, capture_output=True, text=True, encoding="utf-8", errors="replace")
    diem = [l.strip() for l in r.stdout.splitlines() if "ngưỡng" in l or "chạy không lỗi" in l or "TRƯỢT" in l]
    print(f"{d}: {'ĐẠT' if r.returncode == 0 else 'TRƯỢT'} ({(time.time() - t0) / 60:.1f} phút)")
    for l in diem: print("    ", l)
    for pat in RAC:
        for p in (ROOT / d).glob(pat):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
