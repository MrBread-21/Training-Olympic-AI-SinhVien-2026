"""Tạo dữ liệu bài 05 (CPU vài giây).
A: 120 bản đồ lưới có chi phí địa hình, robot ghé mọi điểm mốc rồi về G. Đáp án: chi phí tối ưu.
B: 30 gridworld trơn trượt (mdp.py). Đáp án: giá trị tối ưu và giá trị của chính sách ngẫu nhiên tại S."""
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from mdp import HUONG, chuyen, gia_tri_chinh_sach, thuong

HERE = Path(__file__).resolve().parent
D, A = HERE / "data", HERE / ".dap_an"
D.mkdir(exist_ok=True); A.mkdir(exist_ok=True)
rng = np.random.default_rng(505)


# ------------------------------------------------------------------ A
def ban_do():
    H, W = (int(v) for v in rng.integers(12, 31, 2))
    z = rng.random((H, W))
    for _ in range(3):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    g = (1 + np.floor(z ** 1.3 * 8.999)).astype(int).astype(str).astype(object)
    g[rng.random((H, W)) < 0.1] = "#"
    for _ in range(H * W // 50):
        y, x, L = int(rng.integers(0, H)), int(rng.integers(0, W)), int(rng.integers(3, 10))
        if rng.random() < 0.5: g[y, x:x + L] = "#"
        else: g[y:y + L, x] = "#"
    tu_do = [(y, x) for y in range(H) for x in range(W) if g[y, x] != "#"]
    tp = {}
    for o in tu_do:
        if o in tp: continue
        tp[o] = o; q = deque([o])
        while q:
            y, x = q.popleft()
            for dy, dx in HUONG.values():
                v = (y + dy, x + dx)
                if 0 <= v[0] < H and 0 <= v[1] < W and g[v] != "#" and v not in tp:
                    tp[v] = o; q.append(v)
    lon = pd.Series({k: str(v) for k, v in tp.items()}).value_counts().index[0]
    ung = [o for o in tu_do if str(tp[o]) == lon]
    K = int(rng.integers(4, 10))
    for o, c in zip([ung[i] for i in rng.choice(len(ung), K + 2, replace=False)], ["S", "G"] + [chr(65 + i) for i in range(K)]):
        g[o] = c
    return ["".join(r) for r in g]


def toi_uu(luoi):
    H, W = len(luoi), len(luoi[0])
    vt = {c: (y, x) for y, r in enumerate(luoi) for x, c in enumerate(r) if c in "SG" or c.isalpha() and c.isupper()}

    def dij(s):
        dist, pq = {s: 0}, [(0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]: continue
            for dy, dx in HUONG.values():
                v = (u[0] + dy, u[1] + dx)
                if 0 <= v[0] < H and 0 <= v[1] < W and luoi[v[0]][v[1]] != "#":
                    o = luoi[v[0]][v[1]]; nd = d + (int(o) if o.isdigit() else 1)
                    if nd < dist.get(v, 1 << 30): dist[v] = nd; heapq.heappush(pq, (nd, v))
        return dist
    dd = {c: dij(vt[c]) for c in vt}
    moc = sorted(c for c in vt if c not in "SG"); K = len(moc)
    dp = {(1 << i, i): dd["S"][vt[m]] for i, m in enumerate(moc)}
    for m in range(1, 1 << K):
        for i in range(K):
            if (m, i) not in dp: continue
            for j in range(K):
                if not m >> j & 1:
                    v = dp[(m, i)] + dd[moc[i]][vt[moc[j]]]
                    if v < dp.get((m | 1 << j, j), 1 << 40): dp[(m | 1 << j, j)] = v
    return min(dp[((1 << K) - 1, i)] + dd[moc[i]][vt["G"]] for i in range(K))


maps = [ban_do() for _ in range(120)]
ids = [f"m{i:03d}" for i in range(120)]
pd.DataFrame({"id": ids, "grid": ["/".join(m) for m in maps]}).to_csv(D / "a_maps.csv", index=False)
pd.DataFrame({"id": ids, "optimal_cost": [toi_uu(m) for m in maps]}).to_csv(A / "a_opt.csv", index=False)


# ------------------------------------------------------------------ B
def gridworld():
    H, W = (int(v) for v in rng.integers(5, 10, 2))
    g = np.full((H, W), ".", object)
    g[rng.random((H, W)) < 0.15] = "#"
    o = [(y, x) for y in range(H) for x in range(W) if g[y, x] == "."]
    chon = [o[i] for i in rng.choice(len(o), 2 + int(rng.integers(2, 6)), replace=False)]
    g[chon[0]], g[chon[1]] = "S", "G"
    for c in chon[2:]:
        g[c] = "X"
    return ["".join(r) for r in g]


def toi_uu_mdp(luoi, p, gamma):
    H, W = len(luoi), len(luoi[0])
    V = np.zeros((H, W))
    for _ in range(5000):
        V2 = V.copy()
        for y in range(H):
            for x in range(W):
                if luoi[y][x] not in ".S": continue
                V2[y, x] = max(sum(pr * (thuong(luoi[y2][x2]) + (gamma * V[y2, x2] if luoi[y2][x2] in ".S" else 0))
                                   for pr, y2, x2 in chuyen(luoi, y, x, a, p)) for a in HUONG)
        if np.abs(V2 - V).max() < 1e-10: break
        V = V2
    return V


rows, ans = [], []
while len(rows) < 30:
    luoi = gridworld(); p = round(float(rng.uniform(0.1, 0.35)), 2); gamma = round(float(rng.uniform(0.95, 0.99)), 3)
    H, W = len(luoi), len(luoi[0])
    sy, sx = next((y, r.index("S")) for y, r in enumerate(luoi) if "S" in r)
    V = toi_uu_mdp(luoi, p, gamma)
    if V[sy, sx] < 20:                       # đích quá khó tới: tạo lại
        continue
    ngau = np.mean([gia_tri_chinh_sach(luoi, p, gamma, "".join(rng.choice(list("UDLR"), H * W)))[(sy, sx)] for _ in range(20)])
    i = f"g{len(rows):02d}"
    rows.append((i, "/".join(luoi), p, gamma)); ans.append((i, V[sy, sx], ngau))
pd.DataFrame(rows, columns=["id", "grid", "p", "gamma"]).to_csv(D / "b_mdp.csv", index=False)
pd.DataFrame(ans, columns=["id", "v_opt", "v_rand"]).to_csv(A / "b_opt.csv", index=False)
print("xong: A 120 bản đồ, B 30 gridworld")
