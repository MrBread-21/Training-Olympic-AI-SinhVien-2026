# Lời giải Robot_PathPlanning - Robot đi qua mọi điểm mốc

Đề: [selfhost_kaggle/Robot_PathPlanning](../../selfhost_kaggle/Robot_PathPlanning/). Code: [solve.py](solve.py). Không dùng model pretrained.

## Kết quả (chấm bằng đúng metric của cuộc thi, `selfhost_kaggle/tools/cham.py`)

| | Public | Private |
|---|---|---|
| Baseline của đề | 0,777 | 0,778 |
| **Lời giải này** | 1,000 | 1,000 |

Thời gian chạy: khoảng 30 giây, CPU.

## Cách làm

1. **Dijkstra** từ S, G và từng điểm mốc, với chi phí là giá của ô **bước vào** (đúng luật chấm); giữ con trỏ cha để truy vết đường.
2. **Held-Karp** (quy hoạch động trên tập con): `dp[tập điểm đã ghé][điểm cuối]`, 2^K x K trạng thái, K ≤ 10 nên tức thì.
3. Ghép các đoạn đường S, điểm mốc theo thứ tự tối ưu, G. Kiểm tra trên `train.csv`: 50/50 bản đồ đạt đúng `optimal_cost`.

## Ghi chú

Baseline BFS bỏ qua chi phí địa hình và ghé điểm gần nhất trước, nên mất khoảng 22%. Bài này có lời giải tối ưu chính xác: điểm 1,0 là trần.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py Robot_PathPlanning submission.csv
```
