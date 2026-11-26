# %% [markdown]
# # Baseline - Định vị tàu hoả bốn mùa
#
# Không học gì: thu mỗi ảnh về 64x32 xám, chuẩn hoá từng ô 8x8, so cosine giữa ảnh mùa hè cần định vị và kho ảnh mùa đông,
# lấy ảnh mùa đông giống nhất. Notebook: tự chấm Recall@1 trên validation, rồi ghi `output.csv` (`query_id,winter_id`).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc trước khi chạy.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(42)
if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("train_summer.csv")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)


def vec(p):
    a = np.asarray(Image.open(p).convert("L").resize((64, 32), Image.BILINEAR), dtype=np.float32)
    a = a.reshape(4, 8, 8, 8).transpose(0, 2, 1, 3).reshape(32, -1)  # 32 ô 8x8
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    v = a.flatten()
    return v / np.linalg.norm(v)


def dinh_vi(q_paths, d_paths):
    """Trả về chỉ số ảnh mùa đông giống nhất cho từng ảnh mùa hè."""
    Q = np.stack([vec(p) for p in q_paths]); B = np.stack([vec(p) for p in d_paths])
    return (Q @ B.T).argmax(1)


# %% [markdown]
# ## Tự chấm trên validation (đúng nếu lệch tối đa 10 khung)

# %%
vs, vw = pd.read_csv(D / "validation_summer.csv"), pd.read_csv(D / "validation_winter.csv")
best = dinh_vi([D / f"validation/summer/{i}.jpg" for i in vs.summer_id], [D / f"validation/winter/{i}.jpg" for i in vw.winter_id])
lech = np.abs(vw.position.values[best] - vs.position.values)
print(f"validation Recall@1 = {100 * (lech <= 10).mean():.2f}")

# %% [markdown]
# ## Dự đoán test: test/queries (public) và private_test/queries (nếu đã giải nén)

# %%
db = sorted(p.stem for p in (D / "test/database").glob("*.jpg"))
q_paths = sorted((D / "test/queries").glob("*.jpg"))
for thu in [Path("private_test/queries"), D / "private_test/queries"]:
    if thu.is_dir():
        q_paths += sorted(thu.glob("*.jpg")); break
best = dinh_vi(q_paths, [D / f"test/database/{i}.jpg" for i in db])
out = pd.DataFrame({"query_id": [p.stem for p in q_paths], "winter_id": np.array(db)[best]})
# Kaggle đòi đủ mọi dòng của sample_submission (910): ảnh chưa dự đoán được (private khi chưa có mật khẩu) giữ dòng mẫu
mau = pd.read_csv(D / "sample_submission.csv")
du_doan = dict(zip(out.query_id, out.winter_id))
mau["winter_id"] = [du_doan.get(q, w) for q, w in zip(mau.query_id, mau.winter_id)]
mau.to_csv("output.csv", index=False)
print(f"ghi output.csv: {len(mau)} dòng, {mau.query_id.isin(du_doan).sum()} ảnh mùa hè có dự đoán")
