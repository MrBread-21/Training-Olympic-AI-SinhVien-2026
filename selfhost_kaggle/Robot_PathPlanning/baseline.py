# %% [markdown]
# # Baseline - Robot đi qua mọi điểm mốc rồi về đích
#
# Bỏ qua chi phí địa hình: tìm đường ngắn nhất theo số bước bằng BFS, đi tới điểm mốc gần nhất (theo số bước) chưa ghé,
# hết điểm mốc thì về G. Luôn ra đường hợp lệ nhưng chưa tối ưu. Tự chấm trên train (có optimal_cost),
# rồi ghi `submission.csv` (`id,actions`).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("train.csv")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)
BUOC = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def bfs(luoi, nguon):
    """Đường ít bước nhất từ nguon tới mọi ô: trả về dict ô -> (ô trước, bước)."""
    H, W = len(luoi), len(luoi[0])
    cha, q = {nguon: None}, deque([nguon])
    while q:
        u = q.popleft()
        for c, (dy, dx) in BUOC.items():
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < H and 0 <= v[1] < W and luoi[v[0]][v[1]] != "#" and v not in cha:
                cha[v] = (u, c); q.append(v)
    return cha


def duong_di(cha, dich):
    s, v = "", dich
    while cha[v] is not None:
        v, c = cha[v]; s = c + s
    return s


def giai(grid):
    luoi = grid.split("/")
    vt = {c: (y, x) for y, r in enumerate(luoi) for x, c in enumerate(r) if c in "SG" or "A" <= c <= "L"}
    con = {c for c in vt if c not in "SG"}
    cur, act = vt["S"], ""
    while con:
        cha = bfs(luoi, cur)
        k = min(con, key=lambda c: len(duong_di(cha, vt[c])))
        act += duong_di(cha, vt[k]); cur = vt[k]; con.discard(k)
    return act + duong_di(bfs(luoi, cur), vt["G"])


def chi_phi(grid, act):
    luoi = grid.split("/")
    y, x = next((i, r.index("S")) for i, r in enumerate(luoi) if "S" in r)
    tong = 0
    for c in act:
        y, x = y + BUOC[c][0], x + BUOC[c][1]
        o = luoi[y][x]
        tong += int(o) if o.isdigit() else 1
    return tong


# %% [markdown]
# ## Tự chấm trên train (điểm = optimal_cost / chi phí của mình)

# %%
tr = pd.read_csv(D / "train.csv")
print(f"train: điểm = {np.mean([o / chi_phi(g, giai(g)) for g, o in zip(tr.grid, tr.optimal_cost)]):.4f}")

# %% [markdown]
# ## Dự đoán test (public) và private_test (nếu đã giải nén)

# %%
test = pd.read_csv(D / "test.csv")
for thu in [Path("private_test.csv"), D / "private_test.csv"]:
    if thu.exists():
        test = pd.concat([test, pd.read_csv(thu)]); break
pd.DataFrame({"id": test.id, "actions": [giai(g) for g in test.grid]}).to_csv("submission.csv", index=False)
print(f"ghi submission.csv: {len(test)} bản đồ")
