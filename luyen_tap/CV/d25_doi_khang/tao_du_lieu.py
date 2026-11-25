"""Tạo dữ liệu bài 25: train sẵn CNN nhỏ (mo_hinh.pt) trên 3000 ảnh hình học, chọn 100 ảnh test mô hình đoán đúng.
GPU khoảng 20 giây, CPU khoảng 1-2 phút."""
import sys
from pathlib import Path
import torch                      # import torch trước cv2/pandas (Windows)
import torch.nn.functional as F
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung")); sys.path.insert(0, str(HERE))
from hinh import LOP, ve, luu  # noqa: E402
from mo_hinh import CNN  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(2501)
dev = "cuda" if torch.cuda.is_available() else "cpu"
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "test", A):
    p.mkdir(parents=True, exist_ok=True)

X, y = zip(*[(ve(i % 6, rng)[0], i % 6) for i in range(3000)])
X = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).float() / 255
y = torch.tensor(y)
m = CNN().to(dev)
opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-4)
for ep in range(6):
    m.train()
    for idx in torch.randperm(len(y)).split(64):
        xb = X[idx].to(dev)
        xb = torch.where(torch.rand(len(xb), 1, 1, 1, device=dev) < 0.5, xb.flip(3), xb)
        loss = F.cross_entropy(m(xb), y[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"ep {ep} loss {loss.item():.3f}", flush=True)
m.eval()
torch.save(m.state_dict(), HERE / "mo_hinh.pt")

rows, masks = [], []
while len(rows) < 100:
    k = int(rng.integers(6)); img, mk = ve(k, rng)
    with torch.no_grad():
        p = m(torch.from_numpy(img).permute(2, 0, 1)[None].float().to(dev) / 255).argmax(1).item()
    if p == k:
        i = f"t{len(rows):03d}"; luu(D / "test" / f"{i}.png", img); rows.append((i, LOP[k])); masks.append(mk)
pd.DataFrame(rows, columns=["id", "label"]).to_csv(D / "test.csv", index=False)
pd.DataFrame({"index": range(6), "label": LOP}).to_csv(D / "classes.csv", index=False)   # thứ tự lớp của mo_hinh
np.savez_compressed(A / "masks.npz", ids=np.array([r[0] for r in rows]), masks=np.stack(masks))
print("xong: mo_hinh.pt + 100 ảnh test (lớp theo thứ tự", LOP, ")")
