# %% [markdown]
# # Baseline - Tấn công đối kháng: FGSM một bước lên mô hình A
#
# Với mỗi ảnh test, lấy gradient của loss theo ảnh trên mô hình A (đã công bố) và cộng `8 * sign(gradient)` (thang 0-255),
# làm tròn về số nguyên, kẹp trong [0, 255]. Tự chấm tỉ lệ đánh lừa được mô hình A trên test public (mô hình B bị ẩn,
# chỉ biết điểm của nó qua bảng xếp hạng), rồi ghi `submission.csv` (`id,image`, ảnh mã hoá base64).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import base64
import sys
from pathlib import Path

import torch  # trên Windows torch phải nạp trước pandas
import numpy as np
import pandas as pd
from PIL import Image

if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("model_A.pt")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)
sys.path.insert(0, str(D))
from mo_hinh import tai_mo_hinh  # noqa: E402

EPS = 8  # độ lệch tối đa mỗi điểm ảnh (thang 0-255)
A = tai_mo_hinh(D / "model_A.pt")


def doc(thu_muc_anh, file_nhan):
    nhan = pd.read_csv(file_nhan)
    x = np.stack([np.asarray(Image.open(thu_muc_anh / f"{i}.png").convert("RGB")) for i in nhan.id])
    return nhan.id.tolist(), x, nhan.label.values


def fgsm(model, x, y):
    t = torch.tensor(x).permute(0, 3, 1, 2).float().div(255).requires_grad_(True)
    torch.nn.functional.cross_entropy(model(t), torch.tensor(y)).backward()
    adv = x.astype(np.int16) + EPS * t.grad.sign().permute(0, 2, 3, 1).numpy().astype(np.int16)
    return np.clip(adv, 0, 255).astype(np.uint8)


def doan(model, x):
    with torch.no_grad():
        return model(torch.tensor(x).permute(0, 3, 1, 2).float().div(255)).argmax(1).numpy()


# %% [markdown]
# ## Tấn công test public, tự chấm trên mô hình A

# %%
ids, x, y = doc(D / "test/images", D / "test_labels.csv")
adv = fgsm(A, x, y)
print(f"A đoán đúng ảnh gốc: {np.mean(doan(A, x) == y):.3f}   A bị lừa sau FGSM: {np.mean(doan(A, adv) != y):.3f}")
print("độ lệch lớn nhất:", int(np.abs(adv.astype(int) - x.astype(int)).max()))

# %% [markdown]
# ## Thêm private_test (nếu đã giải nén) và ghi file nộp

# %%
for thu in [Path("private_test"), D / "private_test"]:
    if (thu / "test_labels.csv").exists():
        i2, x2, y2 = doc(thu / "images", thu / "test_labels.csv")
        ids, adv = ids + i2, np.concatenate([adv, fgsm(A, x2, y2)]); break
pd.DataFrame({"id": ids, "image": [base64.b64encode(a.tobytes()).decode() for a in adv]}).to_csv("submission.csv", index=False)
print(f"ghi submission.csv: {len(ids)} ảnh")
