"""Mô hình gridworld trơn trượt dùng chung cho tao_du_lieu.py và check_05.py (thí sinh được đọc file này).

Ô: '#' tường, '.' nền, 'S' xuất phát, 'G' đích (+100, kết thúc), 'X' bẫy (-50, kết thúc).
Mỗi bước: thưởng -1. Hành động U/D/L/R: đi đúng hướng với xác suất 1 - p, trượt sang mỗi hướng vuông góc với xác suất p/2.
Đi vào tường hoặc ra ngoài lưới thì đứng yên. Bước vào G / X nhận thêm thưởng của ô đó và kết thúc. Chiết khấu gamma."""
import numpy as np

HUONG = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
VUONG_GOC = {"U": "LR", "D": "LR", "L": "UD", "R": "UD"}


def chuyen(luoi, y, x, a, p):
    """Danh sách (xác suất, y2, x2) khi làm hành động a ở ô (y, x)."""
    H, W = len(luoi), len(luoi[0])
    out = []
    for h, pr in [(a, 1 - p), (VUONG_GOC[a][0], p / 2), (VUONG_GOC[a][1], p / 2)]:
        y2, x2 = y + HUONG[h][0], x + HUONG[h][1]
        if not (0 <= y2 < H and 0 <= x2 < W) or luoi[y2][x2] == "#":
            y2, x2 = y, x
        out.append((pr, y2, x2))
    return out


def thuong(o):
    return -1 + (100 if o == "G" else -50 if o == "X" else 0)


def gia_tri_chinh_sach(luoi, p, gamma, pol):
    """V^pi chính xác bằng giải hệ tuyến tính. pol: chuỗi H*W ký tự (U/D/L/R; ô tường / kết thúc ký tự gì cũng được)."""
    H, W = len(luoi), len(luoi[0])
    o_song = [(y, x) for y in range(H) for x in range(W) if luoi[y][x] in ".S"]
    chi_so = {o: i for i, o in enumerate(o_song)}
    A, b = np.eye(len(o_song)), np.zeros(len(o_song))
    for (y, x), i in chi_so.items():
        a = pol[y * W + x]
        a = a if a in HUONG else "U"
        for pr, y2, x2 in chuyen(luoi, y, x, a, p):
            b[i] += pr * thuong(luoi[y2][x2])
            if (y2, x2) in chi_so:
                A[i, chi_so[(y2, x2)]] -= gamma * pr
    v = np.linalg.solve(A, b)
    return {o: v[i] for o, i in chi_so.items()}
