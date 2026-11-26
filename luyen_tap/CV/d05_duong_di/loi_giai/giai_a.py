import heapq, random
import numpy as np, pandas as pd

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
seed_all(42)
BUOC = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

def dijkstra(luoi, nguon):                      # chi phí = ô bước vào; giữ cha để truy vết
    H, W = len(luoi), len(luoi[0]); dist, cha, pq = {nguon: 0}, {}, [(0, nguon)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for c, (dy, dx) in BUOC.items():
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < H and 0 <= v[1] < W and luoi[v[0]][v[1]] != "#":
                o = luoi[v[0]][v[1]]; nd = d + (int(o) if o.isdigit() else 1)
                if nd < dist.get(v, float("inf")): dist[v], cha[v] = nd, (u, c); heapq.heappush(pq, (nd, v))
    return dist, cha

def truy_vet(cha, nguon, dich):
    s = []
    while dich != nguon: dich, c = cha[dich]; s.append(c)
    return "".join(reversed(s))

def giai(grid):
    luoi = grid.split("/")
    vt = {c: (y, x) for y, r in enumerate(luoi) for x, c in enumerate(r) if c.isalpha()}
    moc = sorted(c for c in vt if c not in "SG"); K = len(moc)
    dj = {c: dijkstra(luoi, vt[c]) for c in vt}
    d = lambda a, b: dj[a][0][vt[b]]
    INF = float("inf"); dp = [[INF] * K for _ in range(1 << K)]; tr = [[-1] * K for _ in range(1 << K)]
    for i in range(K): dp[1 << i][i] = d("S", moc[i])
    for m in range(1, 1 << K):                  # Held-Karp: dp[tập đã ghé][điểm cuối]
        for i in range(K):
            if dp[m][i] == INF: continue
            for j in range(K):
                if m >> j & 1: continue
                v = dp[m][i] + d(moc[i], moc[j])
                if v < dp[m | 1 << j][j]: dp[m | 1 << j][j], tr[m | 1 << j][j] = v, i
    full = (1 << K) - 1; j = min(range(K), key=lambda i: dp[full][i] + d(moc[i], "G"))
    thu_tu, m = [], full
    while j != -1: thu_tu.append(moc[j]); pj = tr[m][j]; m ^= 1 << j; j = pj
    chuoi = ["S"] + thu_tu[::-1] + ["G"]
    return "".join(truy_vet(dj[a][1], vt[a], vt[b]) for a, b in zip(chuoi, chuoi[1:]))

df = pd.read_csv("data/a_maps.csv")
pd.DataFrame({"id": df.id, "actions": [giai(g) for g in df.grid]}).to_csv("submission_a.csv", index=False)
print("đã ghi submission_a.csv")
