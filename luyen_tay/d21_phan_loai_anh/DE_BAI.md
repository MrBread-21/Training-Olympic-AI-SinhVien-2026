# Bài 21 (21/11) - Phân loại ảnh: khung fine-tune chuẩn

**Thời gian mục tiêu: ≤ 45 phút** (viết + chạy), đóng tài liệu.

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **PyTorch** (đọc theo thứ tự): [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), [Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) (mục `state_dict`), [Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- **Chế độ suy luận**: [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html), mục *Locally disabling gradient computation* (no_grad, inference_mode, `model.eval()` khác nhau thế nào).
- **bf16 / autocast**: [torch.amp](https://pytorch.org/docs/stable/amp.html). **OneCycleLR**: [docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) (`total_steps`, gọi `step()` sau mỗi batch).
- **torchvision**: [Models and pre-trained weights](https://pytorch.org/vision/stable/models.html) (cách gọi `weights="DEFAULT"`, chuẩn hoá ImageNet).
- **Quy trình**: A. Karpathy, [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (đọc hết, ngắn).
- **Bài báo**: He và cộng sự, *Deep Residual Learning* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385), mục 3 (khối residual).
- **Trong máy**: `../../ON_TAP.md` phần 1.1 và phần 3; `../../EuroSAT/solution.ipynb` (tầng 2a: vòng train có ETA).

## Bài 1 - Phân loại hình học

- **Đầu vào**: `data/train/*.png` (64x64 RGB) + `data/train.csv` (`id,label`, 6 lớp: `tron, vuong, tam_giac, chu_thap, vanh, sao`, 200 ảnh/lớp); `data/test/*.png` (600 ảnh, bị xoay/mờ/nhiễu nhẹ); `data/sample_submission.csv`.
- **Đầu ra**: `submission.csv` với cột `id,label`, mỗi id test đúng một dòng.
- **Metric**: macro-F1, **ngưỡng 0.95**.
- **Khung bắt buộc**:
  1. `seed_all(42)`; fp32 trên GTX 1650 (bf16 chỉ khi `get_device_capability()[0] >= 8`).
  2. `Dataset` + `DataLoader` (`num_workers=0`), ảnh HWC sang CHW bằng `permute`, chuẩn hoá ImageNet, nhãn `torch.long`.
  3. Fine-tune `torchvision.models.resnet18(weights="DEFAULT")`, thay `fc`.
  4. Tách val có `stratify`; optimizer + scheduler tạo **một lần**.
  5. Validation **sau mỗi epoch, ngoài vòng batch**; lưu `best.pt` theo val; nạp lại `best.pt` trước khi dự đoán.
  6. Dự đoán theo batch trong `model.eval()` + `torch.inference_mode()`.
  7. Điền file nộp **theo id** (`sub.id.map(...)`), kiểm tra không thiếu id / không NaN.

## Bài 2 - Viết lại nhanh bài 20 (≤ 20 phút)

Viết lại bài 01-08 của `d20_tinh_toan` từ trí nhớ vào `d20_tinh_toan/viet_lai.py` (giữ đúng tên hàm), rồi trong thư mục đó chạy `python kiem_tra.py viet_lai.py 1 2 3 4 5 6 7 8`.

## Cách chạy

```
python tao_du_lieu.py              # tạo data/ (khoảng 10 giây)
python check_21.py bai_lam.py      # chạy bài làm rồi chấm; thêm --khong-chay nếu đã có submission.csv
```

Lời giải mẫu: `loi_giai/giai.py` (chỉ mở sau khi làm xong). Lời giải mẫu đạt macro-F1 0.98 trong 0.5 phút trên GTX 1650.
