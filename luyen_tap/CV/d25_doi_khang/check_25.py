"""Chấm bài 25.  python check_25.py [bai_lam.py] [--khong-chay]"""
import sys
from pathlib import Path
import torch                                                     # import trước cv2 (Windows)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung")); sys.path.insert(0, str(HERE))
from cham import Bang, doi_so, kiem_tra_code, chay  # noqa: E402
from mo_hinh import nap  # noqa: E402
import cv2, numpy as np, pandas as pd  # noqa: E402

NGUONG_TC, NGUONG_PSNR, NGUONG_CAM = 0.55, 30.0, 0.75   # PGD đúng ~0.62, FGSM ~0.42
B = Bang("Bài 25 - tấn công đối kháng + Grad-CAM")
bai, can_chay = doi_so()
kiem_tra_code(B, bai, tu_khoa=[(r"\.sign\(\)", "có bước FGSM/PGD theo dấu gradient (.sign())"),
                               (r"torch\.(min|max|clamp)|clip", "có chiếu về quả cầu L_inf"),
                               (r"autograd\.grad|retain_grad|register_\w*hook|\.grad\b", "Grad-CAM lấy gradient của bản đồ đặc trưng")])
if can_chay:
    chay(B, bai, timeout_phut=10)
te = pd.read_csv(HERE / "data/test.csv"); CLS = pd.read_csv(HERE / "data/classes.csv").label.tolist()
read = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.exists() else None
U8 = np.stack([read(HERE / "data/test" / f"{i}.png") for i in te.id]).astype(int)
adv = [read(bai.parent / "adv" / f"{i}.png") for i in te.id]
if not B.ok(all(a is not None and a.shape == (64, 64, 3) for a in adv), "đủ 100 ảnh adv/<id>.png kích thước 64x64x3"):
    B.ket_thuc()
adv = np.stack(adv).astype(int)
B.ok(np.abs(adv - U8).max() <= 8, f"lệch tối đa so với ảnh gốc = {np.abs(adv - U8).max()} (giới hạn 8 = 8/255)")
psnr = (10 * np.log10(255 ** 2 / np.maximum(((adv - U8) ** 2).mean((1, 2, 3)), 1e-9))).mean()
B.ok(psnr >= NGUONG_PSNR, f"PSNR trung bình = {psnr:.2f} dB (ngưỡng {NGUONG_PSNR})")
dev = "cuda" if torch.cuda.is_available() else "cpu"
m, y = nap(HERE / "mo_hinh.pt", dev), te.label.map(CLS.index).values
with torch.inference_mode():
    p = m(torch.from_numpy(adv).permute(0, 3, 1, 2).float().div(255).to(dev)).argmax(1).cpu().numpy()
B.ok((p != y).mean() >= NGUONG_TC, f"tỉ lệ tấn công thành công sau khi lưu PNG = {(p != y).mean():.2f} (ngưỡng {NGUONG_TC})")
cp = bai.parent / "cam.npy"
if B.ok(cp.exists(), "có cam.npy"):
    cam = np.load(cp)
    B.ok(cam.shape == (100, 64, 64) and np.isfinite(cam).all(), f"cam.npy shape (100, 64, 64), không NaN (đang là {cam.shape})")
    mk = np.load(HERE / ".dap_an/masks.npz")["masks"]
    mk = np.stack([cv2.dilate(k, np.ones((9, 9), np.uint8)) for k in mk])   # cho lệch 4 px = 1 ô của bản đồ 16x16
    hit = np.mean([mk[i].ravel()[cam[i].argmax()] > 0 for i in range(len(mk))])
    B.ok(hit >= NGUONG_CAM, f"pointing game: điểm nóng nhất của CAM nằm trong vật thể (sai số 4 px) ở {hit:.2f} ảnh (ngưỡng {NGUONG_CAM})")
B.ket_thuc()
