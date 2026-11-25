"""Chấm bài 05.  python check_05.py [bai_a.py] [bai_b.py] [--khong-chay]
A: đường đi qua mọi điểm mốc (submission_a.csv: id,actions). B: chính sách cho gridworld trơn trượt (submission_b.csv: id,policy)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chung")); sys.path.insert(0, str(HERE))
from cham import Bang, kiem_tra_code, chay, doc_csv, doc_dap_an  # noqa: E402
from mdp import gia_tri_chinh_sach  # noqa: E402
import pandas as pd  # noqa: E402

NG_A, NG_B = 0.99, 0.97   # lời giải mẫu 1.000 / 1.000; BFS + điểm gần nhất chỉ ~0.78; tham lam một bước ~0.6
args = [a for a in sys.argv[1:] if not a.startswith("--")] + ["bai_a.py", "bai_b.py"][len([a for a in sys.argv[1:] if not a.startswith("--")]):]
fa, fb = [Path(a).resolve() for a in args[:2]]
can_chay = "--khong-chay" not in sys.argv
B = Bang("Bài 05 - tìm đường và MDP")
BUOC = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def diem_duong(grid, act, opt):
    luoi = grid.split("/"); H, W = len(luoi), len(luoi[0])
    act = str(act).strip()
    if any(c not in BUOC for c in act): return 0.0, "ký tự lạ"
    y, x = next((i, r.index("S")) for i, r in enumerate(luoi) if "S" in r)
    con = {c for r in luoi for c in r if c.isalpha() and c not in "SG"}; cost = 0
    for c in act:
        y, x = y + BUOC[c][0], x + BUOC[c][1]
        if not (0 <= y < H and 0 <= x < W) or luoi[y][x] == "#": return 0.0, "ra ngoài / vào tường"
        o = luoi[y][x]; cost += int(o) if o.isdigit() else 1; con.discard(o)
    if con: return 0.0, "thiếu điểm mốc"
    if luoi[y][x] != "G": return 0.0, "không kết thúc ở G"
    return min(1.0, opt / cost), ""


print("A - Dijkstra + Held-Karp")
kiem_tra_code(B, fa, torch=False, tu_khoa=[(r"heapq", "dùng heapq (Dijkstra)"), (r"1\s*<<", "quy hoạch động trên tập con (bitmask)")])
if can_chay: chay(B, fa, timeout_phut=5)
ans = doc_dap_an(HERE, "a_opt.csv"); maps = pd.read_csv(HERE / "data/a_maps.csv").set_index("id")
d = doc_csv(B, fa.parent / "submission_a.csv", ["id", "actions"], ans.id)
kq = [diem_duong(maps.grid[i], d.actions[i], o) for i, o in zip(ans.id, ans.optimal_cost)]
loi = pd.Series([e for _, e in kq if e]).value_counts().to_dict()
s = np.mean([k for k, _ in kq])
B.ok(s >= NG_A, f"A: điểm = {s:.4f} (ngưỡng {NG_A}); đường không hợp lệ: {loi or 'không có'}")

print("\nB - value iteration")
kiem_tra_code(B, fb, torch=False, tu_khoa=[(r"gamma", "có chiết khấu gamma"), (r"max\(", "lấy max theo hành động (phương trình Bellman)")])
if can_chay: chay(B, fb, timeout_phut=5)
ans = doc_dap_an(HERE, "b_opt.csv"); mdp = pd.read_csv(HERE / "data/b_mdp.csv").set_index("id")
d = doc_csv(B, fb.parent / "submission_b.csv", ["id", "policy"], ans.id)
diem = []
for i, vo, vr in zip(ans.id, ans.v_opt, ans.v_rand):
    luoi = mdp.grid[i].split("/"); pol = str(d.policy[i])
    if len(pol) != len(luoi) * len(luoi[0]):
        diem.append(0.0); continue
    sy, sx = next((y, r.index("S")) for y, r in enumerate(luoi) if "S" in r)
    v = gia_tri_chinh_sach(luoi, mdp.p[i], mdp.gamma[i], pol)[(sy, sx)]
    diem.append(float(np.clip((v - vr) / (vo - vr), 0, 1)))
s = np.mean(diem)
B.ok(s >= NG_B, f"B: điểm = {s:.4f} (ngưỡng {NG_B}; điểm = (V_pi(S) - V_ngẫu_nhiên) / (V*(S) - V_ngẫu_nhiên))")
B.ket_thuc()
