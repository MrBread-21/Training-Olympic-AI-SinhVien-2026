# Bài 05 (5/12) - Tìm đường qua các điểm mốc, value iteration trên gridworld trơn trượt

**Thời gian mục tiêu: ≤ 45 phút** (A 25, B 20). Không cần GPU.

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính. Lúc làm bài thì đóng hết.

- **Bài học**: [on_tap/dang_bai/16_do_thi_robot.md](../../../on_tap/dang_bai/16_do_thi_robot.md).
- **Python**: [heapq](https://docs.python.org/3/library/heapq.html) (hàng đợi ưu tiên cho Dijkstra).
- **Thuật toán**: Wikipedia [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm), [Held-Karp algorithm](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm) (quy hoạch động trên tập con, O(2^K K^2)).
- **MDP**: Sutton và Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), mục 4.4 (value iteration) và 3.5-3.6 (phương trình Bellman).
- **Trong máy**: [mdp.py](mdp.py) mô tả chính xác luật chuyển trạng thái và phần thưởng của phần B (được đọc).

## Phần A - Robot ghé mọi điểm mốc (`bai_a.py`)

- **Đầu vào**: `data/a_maps.csv` (`id,grid`), 120 bản đồ 12-30 x 12-30. `grid` là các hàng nối bằng `/`. Ô: `#` tường; `1`-`9` chi phí khi bước vào; `S` xuất phát; `G` đích; `A`, `B`, ... điểm mốc (4-9 điểm), bước vào tốn 1.
- **Đầu ra**: `submission_a.csv` (`id,actions`): chuỗi `U/D/L/R` đi từ S, ghé mọi điểm mốc (thứ tự tuỳ ý), kết thúc ở G.
- **Metric**: trung bình `chi phí tối ưu / chi phí của bạn` (đường không hợp lệ được 0). **Ngưỡng 0.99**. BFS theo số bước + ghé điểm gần nhất chỉ được khoảng 0.79.
- **Khung**: Dijkstra từ mỗi điểm then chốt (giữ con trỏ cha để truy vết đường); Held-Karp chọn thứ tự ghé; ghép các đoạn đường.

## Phần B - Chính sách tối ưu cho gridworld trơn trượt (`bai_b.py`)

- **Đầu vào**: `data/b_mdp.csv` (`id,grid,p,gamma`), 30 lưới 5-9 x 5-9. Ô: `#` tường, `.` nền, `S` xuất phát, `G` đích (+100, kết thúc), `X` bẫy (-50, kết thúc). Mỗi bước thưởng -1. Hành động đi đúng hướng với xác suất `1 - p`, trượt sang mỗi hướng vuông góc với xác suất `p/2`; đụng tường hay ra ngoài thì đứng yên. Chiết khấu `gamma`. (Chi tiết: `mdp.py`.)
- **Đầu ra**: `submission_b.csv` (`id,policy`): chuỗi dài `H x W` ký tự `U/D/L/R` theo thứ tự hàng (ô tường và ô kết thúc ghi ký tự gì cũng được).
- **Metric**: `(V_pi(S) - V_ngẫu_nhiên(S)) / (V*(S) - V_ngẫu_nhiên(S))`, trung bình. **Ngưỡng 0.97**. Chính sách tối ưu khi **bỏ qua trượt** (p = 0) chỉ được khoảng 0.93: nó đi sát bẫy.
- **Khung**: value iteration `V(s) = max_a tổng P(s'|s,a) [r + gamma V(s')]` (ô kết thúc không cộng giá trị tương lai), lặp tới khi thay đổi < 1e-9; chính sách = argmax theo Q.

## Bài 2 - Viết lại nhanh bài 04 phần A (≤ 25 phút)

Trong `d04_dimabsa`: `viet_lai_a.py` (hồi quy VA cho khía cạnh), chấm `python check_04.py viet_lai_a.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_05.py bai_a.py bai_b.py
```

## Mở rộng (không chấm)

- Q-learning trên cùng gridworld: viết hàm `step(s, a)` mô phỏng theo `mdp.py`, train 20.000 tập, so điểm với value iteration bằng `mdp.gia_tri_chinh_sach`.
- A* với heuristic Manhattan cho phần A, đo số ô đã duyệt so với Dijkstra.
