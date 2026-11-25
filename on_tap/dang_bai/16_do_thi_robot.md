# Dạng 16 - Tìm kiếm, lập kế hoạch đường đi, học tăng cường

## 1. Nhận dạng đề

- Lưới / đồ thị có vật cản và chi phí; tìm đường cho robot qua các điểm mốc (Robot_PathPlanning tự host), lập lịch, bài toán tối ưu tổ hợp nhỏ.
- Biến thể học tăng cường: môi trường gridworld có phần thưởng, học chính sách bằng Q-learning; môi trường có ngẫu nhiên (trượt chân).
- Không cần GPU; điểm đến từ **thuật toán đúng** và cài đặt không lỗi.

## 2. Metric

Thường là tỉ lệ `chi phí tối ưu / chi phí của bạn` (đường không hợp lệ được 0), hoặc tổng phần thưởng. **Kiểm tra tính hợp lệ** trước khi tối ưu.

## 3. Lộ trình

| Bước | Việc | Kỳ vọng (Robot_PathPlanning) |
|---|---|---|
| 0 | Baseline: BFS theo số bước, ghé điểm gần nhất trước | 0,777 |
| 1 | **Dijkstra** theo chi phí thật (chi phí bước vào ô) giữa mọi cặp điểm then chốt | tăng |
| 2 | Thứ tự ghé tối ưu: **quy hoạch động trên tập con** (≤ 12-15 điểm) | 1,0 (tối ưu) |
| 3 | Nhiều điểm (20+): láng giềng gần nhất + 2-opt / or-opt cải tiến cục bộ | gần tối ưu |

## 4. Khung code

Dijkstra trên lưới (chi phí bước vào ô), có truy vết:

```python
import heapq
BUOC = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
def dijkstra(luoi, nguon):
    H, W = len(luoi), len(luoi[0]); dist, cha = {nguon: 0}, {}; pq = [(0, nguon)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for c, (dy, dx) in BUOC.items():
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < H and 0 <= v[1] < W and luoi[v[0]][v[1]] != "#":
                o = luoi[v[0]][v[1]]; nd = d + (int(o) if o.isdigit() else 1)
                if nd < dist.get(v, float("inf")):
                    dist[v], cha[v] = nd, (u, c); heapq.heappush(pq, (nd, v))
    return dist, cha

def truy_vet(cha, nguon, dich):
    s = []
    while dich != nguon: dich, c = cha[dich]; s.append(c)
    return "".join(reversed(s))
```

Quy hoạch động trên tập con (Held-Karp), bắt đầu ở S, ghé đủ K điểm, kết thúc ở G:

```python
def held_karp(d, K):                   # d[i][j]: chi phí ngắn nhất; chỉ số 0..K-1 là điểm mốc, K là S, K+1 là G
    INF = float("inf"); dp = [[INF] * K for _ in range(1 << K)]; tr = [[-1] * K for _ in range(1 << K)]
    for i in range(K): dp[1 << i][i] = d[K][i]
    for m in range(1, 1 << K):
        for i in range(K):
            if dp[m][i] == INF or not m >> i & 1: continue
            for j in range(K):
                if m >> j & 1: continue
                v = dp[m][i] + d[i][j]
                if v < dp[m | 1 << j][j]: dp[m | 1 << j][j], tr[m | 1 << j][j] = v, i
    full = (1 << K) - 1; j = min(range(K), key=lambda i: dp[full][i] + d[i][K + 1])
    best, order, m = dp[full][j] + d[j][K + 1], [], full
    while j != -1: order.append(j); pj = tr[m][j]; m ^= 1 << j; j = pj
    return best, order[::-1]
```

A* (khi chỉ cần một cặp, lưới lớn): như Dijkstra nhưng ưu tiên `d + h(v)`, với `h` = khoảng cách Manhattan x chi phí ô nhỏ nhất (bằng 1 ở đây) để không đánh giá quá.

Q-learning trên gridworld:

```python
Q = np.zeros((n_trang_thai, n_hanh_dong)); eps, alpha, gamma = 1.0, 0.1, 0.99
for tap in range(5000):
    s = env.reset(); xong = False
    while not xong:
        a = np.random.randint(n_hanh_dong) if np.random.rand() < eps else int(Q[s].argmax())
        s2, r, xong = env.step(a)
        Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not xong) - Q[s, a])
        s = s2
    eps = max(0.05, eps * 0.999)
```

Môi trường biết trước xác suất chuyển: dùng **value iteration** (chính xác, nhanh hơn Q-learning):
`V(s) = max_a tổng_s' P(s'|s,a) [r + gamma V(s')]`, lặp tới khi thay đổi < 1e-6.

## 5. Nút vặn

1. Mô hình hoá chi phí đúng đề (chi phí bước vào ô hay rời ô, ô đặc biệt).
2. Held-Karp khi số điểm nhỏ; 2-opt khi lớn.
3. Kiểm tra lời giải bằng hàm chấm tự viết trên dữ liệu có đáp án trước khi nộp.

## 6. Bẫy

- Nhầm trục (hàng = y, cột = x) và hướng U / D.
- Đường đi qua G trước khi ghé đủ điểm rồi "kết thúc sớm": chỉ vị trí cuối cùng mới tính.
- Đệ quy sâu trong Python (DFS trên lưới lớn) tràn stack: dùng vòng lặp / hàng đợi.

## 7. Bài luyện

- Code tay: [d05_duong_di](../../luyen_tap/CV/d05_duong_di/) (Dijkstra, Held-Karp, Q-learning và value iteration trên gridworld trơn trượt).
- Thi thử: [Robot_PathPlanning](../../selfhost_kaggle/Robot_PathPlanning/) (baseline 0,777; tối ưu 1,0).

## 8. Đọc thêm

- Russell và Norvig, *Artificial Intelligence: A Modern Approach*, chương 3 (tìm kiếm, A*) và chương 17, 22 (MDP, học tăng cường).
- Sutton và Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), chương 4 (quy hoạch động) và 6 (Q-learning).
