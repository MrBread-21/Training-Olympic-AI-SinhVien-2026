# Robot đi qua mọi điểm mốc (lập kế hoạch đường đi)

Dạng bài **lập kế hoạch chuyển động cho robot** trên lưới: robot xuất phát ở S, phải ghé mọi điểm mốc (theo thứ tự tuỳ chọn) rồi dừng ở đích G, với tổng chi phí địa hình nhỏ nhất. Kiến thức cần: BFS, Dijkstra / A*, quy hoạch động trên tập con (bài toán người du lịch nhỏ), hoặc tìm kiếm cục bộ.

## Overview

Mỗi bản đồ là một lưới chữ nhật (20-48 hàng, 20-48 cột):

- `#`: tường, không đi vào được.
- `1`-`9`: ô đất, bước vào ô tốn đúng chừng ấy chi phí (bùn, cát, đồi...).
- `S`: điểm xuất phát. `G`: đích. `A`, `B`, `C`...: điểm mốc (3 đến 10 điểm mỗi bản đồ). Bước vào các ô này tốn 1.

Robot đi 4 hướng: `U` (lên), `D` (xuống), `L` (trái), `R` (phải). Chi phí của một đường đi là tổng chi phí các ô robot bước vào (không tính ô S lúc đầu).

**Nhiệm vụ**: với mỗi bản đồ test, đưa ra chuỗi bước đi từ S, ghé mọi điểm mốc ít nhất một lần, kết thúc ở G. Được đi qua G hay một điểm mốc nhiều lần; chỉ vị trí cuối cùng phải là G.

**Luật**:

- Được dùng mọi thuật toán và thư viện có sẵn. Không sửa tay đường đi cho test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.

## Evaluation

Mỗi bản đồ:

- Đường **hợp lệ** (không ra ngoài lưới, không đi vào tường, ghé đủ mọi điểm mốc, kết thúc ở G): điểm = `chi phí tối ưu / chi phí của bạn`, tối đa 1.
- Không hợp lệ, chuỗi có ký tự khác `UDLR`, hoặc dài hơn 4 x số ô của lưới: điểm 0.

Điểm cuối là trung bình trên mọi bản đồ, thang 0-1. Chi phí tối ưu được tính bằng Dijkstra giữa các điểm then chốt cộng quy hoạch động trên thứ tự ghé điểm mốc.

Ví dụ (lưới 3 x 4, một điểm mốc A):

```
S19G
1A1#
1111
```

G chỉ vào được từ ô `9` bên trái (bên dưới G là tường), nên mọi đường hợp lệ đều phải trả 9 cho ô đó. Chi phí tối ưu là 13:

| Đường | Các ô bước vào | Chi phí | Điểm |
|---|---|---|---|
| `DRRUR` | 1, A, 1, 9, G | 1 + 1 + 1 + 9 + 1 = 13 | 13 / 13 = 1 |
| `DRDRUUR` | 1, A, 1, 1, 1, 9, G | 15 | 13 / 15 = 0,867 |
| `RRR` | 1, 9, G | không ghé A | 0 |
| `DRRURD` | ..., G, rồi đi xuống tường | đi vào tường | 0 |

## Submission File

`submission.csv` có tiêu đề `id,actions`, mỗi bản đồ test một dòng:

```
id,actions
te012345,RRDDDLURRRRDD
```

Public là 500 bản đồ trong `test.csv`, private là 500 bản đồ trong `private_test.zip`. Nên nộp đủ 1.000 dòng; trước khi có mật khẩu, giữ nguyên các dòng private như `sample_submission.csv`.

## Data

```
train.csv              300 bản đồ kèm lời giải tối ưu: id, grid, optimal_cost, actions
test.csv               500 bản đồ public: id, grid
private_test.zip       (có mật khẩu) private_test.csv: 500 bản đồ private
sample_submission.csv  id,actions (để trống, được 0)
baseline.ipynb, baseline.py
```

`grid` là các hàng của lưới nối bằng dấu `/`, ví dụ `S19G/1A1#/1111`. Đọc bằng `grid.split("/")`.

Giải nén private test (khi có mật khẩu):

```python
import zipfile
zipfile.ZipFile("private_test.zip").extractall(".", pwd=b"<mật khẩu>")   # hoặc: unzip -P <mật khẩu> private_test.zip
```

## Hướng làm gợi ý

- Baseline (BFS theo số bước, ghé điểm gần nhất trước) bỏ qua chi phí địa hình và thứ tự ghé tốt.
- Dijkstra (chi phí bước vào ô) giữa mọi cặp điểm then chốt, rồi chọn thứ tự ghé tốt nhất: với tối đa 10 điểm mốc, quy hoạch động trên tập con (2^10 x 10 trạng thái) cho lời giải tối ưu.
- Kiểm tra lời giải trên `train.csv` (có `optimal_cost`) trước khi nộp.

Bản đồ do người dựng đề sinh ngẫu nhiên (địa hình nhiễu làm mượt, tường rải rác và các đoạn tường dài).
