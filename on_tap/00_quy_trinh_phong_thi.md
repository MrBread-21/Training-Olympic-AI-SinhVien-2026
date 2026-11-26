# 00 - Quy trình phòng thi 6 tiếng

Tài liệu này trả lời: vào phòng thi thì làm gì theo từng phút, chia việc ra sao, nộp bài thế nào để không mất điểm vì lỗi thủ tục. Đọc lại ngày 6/12.

## 1. Luật cần thuộc (theo hướng dẫn 2025 và 2026)

| Mục | Nội dung | Hệ quả khi làm bài |
|---|---|---|
| Đội | 2-3 người, 2 máy, 1 tài khoản GPU cloud | GPU dùng chung: một người giữ "lịch GPU", không ai chạy song song hai job nặng |
| Mạng | Không Internet, không `pip install`, không tải model | Mọi thứ phải có sẵn trong môi trường; thuộc khung code |
| Pretrained | CV: chỉ backbone ImageNet trong danh sách BTC. NLP: không pretrained nào | Đọc kỹ luật **từng bài**, có bài cấm hẳn pretrained |
| Dữ liệu | Chỉ dữ liệu BTC cấp; được augmentation và dữ liệu tổng hợp | Không dùng test để pseudo-label nếu luật không cho |
| Nộp bài | Mỗi bài tối đa 20 lần public trong giờ 0-5; giờ 6 mở `private_test` bằng mật khẩu, tối đa 5 lần | Phải có hàm `predict(thư_mục)` chạy nhanh trên dữ liệu mới |
| Hợp lệ | Mỗi lần nộp: zip gồm `output.csv` và `.ipynb` chạy từ trên xuống ra đúng file đó | Notebook phải sạch, chạy lại được; seed cố định |
| Tái lập | `Final/Tac_vu_k/main.py` sinh lại kết quả tốt nhất trong ≤ 20 phút | Tách train và suy luận từ đầu, lưu weights |
| Điểm | Mỗi bài: `(điểm - Min) / (Max - Min)`, Min là baseline BTC, Max là mô hình mạnh của BTC; điểm cuối là trung bình các bài | Bài yếu nhất kéo điểm xuống nhiều nhất: đưa **mọi bài vượt baseline** trước |
| LLM | 2025: có LLM offline (deepseek-r1-distill-qwen-32b) trong 5 giờ đầu, giới hạn token | Tập nhờ LLM yếu viết đoạn code nhỏ, có kiểm tra |

## 2. Chia vai đội 3 người

- **A**: bài NLP (hoặc bài thứ nhất).
- **B**: bài CV (hoặc bài thứ hai).
- **C**: giữ GPU và sổ nộp bài, dựng `Final/`, viết báo cáo, hỗ trợ bài đang kém nhất (tính theo điểm quy đổi, không theo điểm thô).

Đội 2 người: mỗi người một bài, việc của C chia đôi; người nào xong baseline trước giữ sổ nộp bài.

## 3. Lịch theo phút

| Giờ | Việc | Lượt nộp mỗi bài |
|---|---|---|
| 0:00-0:20 | Đọc hết đề (cả các bài không phải của mình). Chạy notebook baseline của BTC, nộp ngay để kiểm tra định dạng | 1 |
| 0:20-1:30 | EDA 15 phút (phân bố nhãn, độ dài, kích thước ảnh, khác biệt train/test), dựng val **giống test**, bản cải tiến đầu tiên vượt baseline | +2-3 |
| 1:30-4:00 | Vòng lặp cải tiến: phần nặng (train backbone, trích đặc trưng) chạy **một lần** rồi lưu; mỗi lượt nộp là một biến thể rẻ (head, ngưỡng, trọng số ensemble) | khoảng 1 lượt / 15 phút, tổng 12-15 |
| 4:00-4:45 | Chốt cấu hình tốt nhất: train bản cuối (train + val), lưu weights, viết `main.py`, chạy lại **hai lần** xem có ra cùng file | giữ 3-5 lượt dự phòng |
| 4:45-5:00 | Dự phòng; viết nháp báo cáo | |
| 5:00-6:00 | Nhận mật khẩu, giải nén, chạy `predict`. 5 lượt: bản tốt nhất trước, rồi các biến thể ensemble. Hoàn tất `Final/`, nén zip, kiểm tra lần cuối | ≤ 5 |

Quy tắc dừng: một hướng đã tốn 45 phút mà val không tăng thì bỏ, quay về bản tốt nhất.

## 4. Notebook khung (8 khối, gõ được trong 15 phút)

```python
# 1. seed + thiết bị
import os, random, time, json, numpy as np, torch
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_all(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8   # Ampere trở lên (A100, L4, RTX 30xx): bf16; GTX 16xx (7.5) thì fp32

# 2. dữ liệu: Dataset/DataLoader, hoặc nạp hết lên GPU nếu vừa
# 3. make_model(): kiến trúc weights=None hoặc backbone ImageNet được phép
# 4. train(): AdamW + OneCycle, in ETA sau 20 batch đầu, lưu best theo val
# 5. predict(model, items, tta=True) -> xác suất / điểm
# 6. metric tự viết, kiểm tra bằng sklearn khi có
# 7. write_submission(pred, tag): ghi output.csv + bản sao outputs/<tag>.csv + dòng vào so_nop_bai.csv
# 8. main.py: nạp weights đã lưu, predict trên thư mục test truyền vào, ghi output.csv
```

In ETA sau vài batch, ví dụ:

```python
t0 = time.time()
for step, (x, y) in enumerate(loader):
    ...
    if step == 20:
        per = (time.time() - t0) / 21
        print(f"{per:.3f}s/batch, 1 epoch ~ {per * len(loader) / 60:.1f} phút, {EPOCHS} epoch ~ {per * len(loader) * EPOCHS / 60:.0f} phút")
```

## 5. Sổ nộp bài (bắt buộc)

Một file `so_nop_bai.csv` cho mỗi bài:

| lần | giờ | tag | thay đổi so với lần trước | điểm val | điểm public | ghi chú |
|---|---|---|---|---|---|---|
| 1 | 0:15 | baseline | notebook BTC | 0.61 | 0.60 | định dạng OK |

Lý do: biết val có đáng tin không (cột val và public cùng tăng?), biết lượt nào để chọn ở giờ 6, và có số liệu cho báo cáo.

## 6. `main.py` chuẩn

```python
"""python main.py <thư_mục_test> <file_ra>: nạp weights đã lưu, dự đoán, ghi file nộp. Không train lại."""
import sys
from pathlib import Path
import torch
from common import seed_all, make_model, predict, write_csv   # tách hàm dùng chung ra file riêng

seed_all(42)
test_dir, out = Path(sys.argv[1]), sys.argv[2]
model = make_model(); model.load_state_dict(torch.load("weights/best.pt", map_location="cpu")); model.eval()
write_csv(predict(model, test_dir), out)
```

Kiểm tra trước giờ 6: chạy `main.py` trên thư mục test public, so với `output.csv` đã nộp (phải giống hệt, hoặc lệch dưới 1e-6 nếu dùng GPU).

## 7. Checklist trước mỗi lượt nộp

- [ ] Đúng tên file (`output.csv`), đúng tên và thứ tự cột, đúng kiểu id (đọc id bằng `dtype=str` để không mất số 0 ở đầu).
- [ ] Đủ số dòng, không id thừa, không id lặp, không NaN.
- [ ] Notebook chạy từ trên xuống ra đúng file này (Restart and Run All nếu còn thời gian; tối thiểu: không có ô chạy lẻ sau khi ghi file).
- [ ] Ghi dòng vào sổ nộp bài.

## 8. Checklist giờ thứ 6

- [ ] Giải nén private đúng chỗ (`zipfile.ZipFile(p).extractall(".", pwd=b"...")` hoặc `unzip -P`).
- [ ] Chạy `predict` cho private, ghép với public nếu file nộp cần cả hai.
- [ ] Lượt 1: bản tốt nhất đã kiểm chứng. Lượt 2-4: ensemble / biến thể. Lượt 5: để dành.
- [ ] `Final/Tac_vu_k/`: `main.py`, weights, notebook, `output.csv` tốt nhất, README ngắn (cách chạy, thời gian).
- [ ] Báo cáo: phương pháp, bảng sổ nộp bài, điều đã thử không hiệu quả.

## 9. Lỗi thủ tục từng gặp (đừng lặp lại)

- Id toàn chữ số bị pandas đọc thành số, mất số 0 ở đầu: luôn `pd.read_csv(..., dtype={"id": str})`.
- Windows: `import torch` phải đứng trước `pandas`/`cv2`/`pyarrow`, nếu không lỗi DLL `c10.dll`. TensorFlow phải nạp trước pandas.
- `num_workers > 0` trên Windows trong notebook hay treo: để 0 khi luyện ở nhà; máy thi là Linux thì dùng 4-8.
- GTX 1650: AMP fp16 chậm hơn fp32 khoảng 3 lần; `channels_last` không có lợi; augmentation trên GPU xong phải `.contiguous()` (nhanh gấp 4).
- Ổ đĩa chậm: đọc nhiều file nhỏ mất khoảng 10 ms/file. Nạp một lần vào RAM (zip bytes hoặc `.npy`), giải mã JPEG trên GPU (`torchvision.io.decode_jpeg(device="cuda")`).
- Chạy hai job GPU cùng lúc làm cả hai chậm và dễ hết VRAM: kiểm tra `nvidia-smi` trước khi chạy.
