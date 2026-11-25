import random
import numpy as np, pandas as pd

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
seed_all(42)
HUONG = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
VUONG_GOC = {"U": "LR", "D": "LR", "L": "UD", "R": "UD"}

def chuyen(luoi, y, x, a, p):                   # đúng hướng 1-p, trượt vuông góc p/2 mỗi bên, đụng tường thì đứng yên
    H, W = len(luoi), len(luoi[0]); out = []
    for h, pr in [(a, 1 - p), (VUONG_GOC[a][0], p / 2), (VUONG_GOC[a][1], p / 2)]:
        y2, x2 = y + HUONG[h][0], x + HUONG[h][1]
        if not (0 <= y2 < H and 0 <= x2 < W) or luoi[y2][x2] == "#": y2, x2 = y, x
        out.append((pr, y2, x2))
    return out

def q_val(luoi, V, y, x, a, p, gamma):
    s = 0.0
    for pr, y2, x2 in chuyen(luoi, y, x, a, p):
        o = luoi[y2][x2]
        r = -1 + (100 if o == "G" else -50 if o == "X" else 0)
        s += pr * (r + (gamma * V[y2, x2] if o in ".S" else 0.0))     # G, X kết thúc: không cộng giá trị tương lai
    return s

def value_iteration(luoi, p, gamma, tol=1e-9):
    H, W = len(luoi), len(luoi[0]); V = np.zeros((H, W))
    while True:
        V2 = V.copy()
        for y in range(H):
            for x in range(W):
                if luoi[y][x] in ".S": V2[y, x] = max(q_val(luoi, V, y, x, a, p, gamma) for a in HUONG)
        if np.abs(V2 - V).max() < tol: return V2
        V = V2

def chinh_sach(luoi, p, gamma):
    V = value_iteration(luoi, p, gamma); H, W = len(luoi), len(luoi[0])
    return "".join(max(HUONG, key=lambda a: q_val(luoi, V, y, x, a, p, gamma)) if luoi[y][x] in ".S" else "U"
                   for y in range(H) for x in range(W))

df = pd.read_csv("data/b_mdp.csv")
pd.DataFrame({"id": df.id, "policy": [chinh_sach(g.split("/"), p, gm) for g, p, gm in zip(df.grid, df.p, df.gamma)]}).to_csv("submission_b.csv", index=False)
print("đã ghi submission_b.csv")
