# Robot_PathPlanning - nội dung từng ô trên Kaggle

## Title

OlpAI thử - Robot đi qua mọi điểm mốc

## Competition URL

pka-olpai-test-robot-path

## Subtitle

Tìm đường rẻ nhất cho robot trên lưới có địa hình và tường: ghé mọi điểm mốc rồi về đích.

## Overview

Robot xuất phát ở S, ghé mọi điểm mốc theo thứ tự tuỳ chọn rồi dừng ở G, tổng chi phí địa hình nhỏ nhất. Điểm mỗi bản đồ = chi phí tối ưu / chi phí của bạn.

## Description

Dạng bài **lập kế hoạch chuyển động cho robot** trên lưới: robot xuất phát ở S, phải ghé mọi điểm mốc (theo thứ tự tuỳ chọn) rồi dừng ở đích G, với tổng chi phí địa hình nhỏ nhất. Kiến thức cần: BFS, Dijkstra / A*, quy hoạch động trên tập con (bài toán người du lịch nhỏ), hoặc tìm kiếm cục bộ.

Mỗi bản đồ là một lưới chữ nhật (20-48 hàng, 20-48 cột):

- `#`: tường, không đi vào được.
- `1`-`9`: ô đất, bước vào ô tốn đúng chừng ấy chi phí (bùn, cát, đồi...).
- `S`: điểm xuất phát. `G`: đích. `A`, `B`, `C`...: điểm mốc (3 đến 10 điểm mỗi bản đồ). Bước vào các ô này tốn 1.

Robot đi 4 hướng: `U` (lên), `D` (xuống), `L` (trái), `R` (phải). Chi phí của một đường đi là tổng chi phí các ô robot bước vào (không tính ô S lúc đầu).

**Nhiệm vụ**: với mỗi bản đồ test, đưa ra chuỗi bước đi từ S, ghé mọi điểm mốc ít nhất một lần, kết thúc ở G. Được đi qua G hay một điểm mốc nhiều lần; chỉ vị trí cuối cùng phải là G.

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

## Timeline

- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó.

## Rules

- Được dùng mọi thuật toán và thư viện có sẵn. Không sửa tay đường đi cho test.
- 5 tiếng đầu: tối đa 20 lần nộp public. Tiếng cuối: mật khẩu `private_test.zip` được công bố, tối đa 5 lần nộp private.
- Cuối giờ phải có `main.py` sinh lại file nộp tốt nhất trong dưới 20 phút, seed cố định.
- Không mở `private_test.zip` trước khi mật khẩu được công bố.
- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private.

## Citation

Bản đồ sinh ngẫu nhiên cho buổi luyện tập OlpAI'26.

## Dataset Description

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

## Data license

CC0: bản đồ do người dựng đề sinh ngẫu nhiên.

## Files để tải lên

- Data: `Robot_PathPlanning_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `path_score`), cao hơn là tốt hơn
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` (nếu có) phải ra điểm tối đa; nộp `submission.csv` của baseline phải gần `host/min_score.txt` (public 0.7770, private 0.7777)
