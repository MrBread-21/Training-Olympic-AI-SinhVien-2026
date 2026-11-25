# 02 - Kỹ thuật huấn luyện và nâng điểm (dùng cho mọi dạng bài)

## 1. Chẩn đoán trước khi sửa

| Triệu chứng | Nghĩa là | Làm gì |
|---|---|---|
| Train loss cao, val cũng kém | Underfit | Train lâu hơn, lr hợp lý hơn, mô hình to hơn, bớt regularization, kiểm tra dữ liệu và nhãn có khớp |
| Train tốt, val kém | Overfit | Augmentation mạnh và đúng kiểu test, weight decay, dropout, MixUp / CutMix, label smoothing, mô hình nhỏ hơn, dùng đặc trưng đóng băng |
| Loss NaN | lr quá cao, fp16 tràn, dữ liệu có NaN | Giảm lr, `clip_grad_norm_(1.0)`, fp32, `assert not np.isnan(x).any()` |
| Loss không giảm ngay từ đầu | Sai nhãn / sai thứ tự trục / quên `zero_grad` / lr quá nhỏ | Overfit thử 1 batch: loss phải về gần 0 sau vài trăm bước |
| Val tốt, LB kém | Val không giống test, hoặc chọn mô hình quá khớp val | Dựng val mô phỏng test (làm hỏng giống test, chia theo nhóm), tin LB hơn khi val nhỏ |
| Val dao động mạnh giữa các lần chạy | Val quá nhỏ | K-fold, trung bình nhiều seed, không tin chênh lệch nhỏ |

Luôn xem **ma trận nhầm lẫn** và **20 mẫu sai nặng nhất** trước khi đổi mô hình: thường lộ ra lỗi dữ liệu, lớp hay nhầm, hay biến đổi test mà train không có.

## 2. Dựng validation

- Val phải giống test: cùng kiểu biến đổi (ảnh test bị hỏng thì val cũng làm hỏng: bài EuroSAT val sạch 0,895 nhưng LB chỉ 0,70, val hỏng 0,68 sát LB), cùng cách chia.
- Dữ liệu có nhóm (video, người nói, bệnh nhân, bài Wikipedia, cảnh camera): `GroupKFold` / `StratifiedGroupKFold` theo nhóm. Bài 23 luyện tay: chia ngẫu nhiên cao hơn thật 0,06.
- Dữ liệu theo thời gian: val là đoạn cuối của train (bài Home Credit).
- Không rò rỉ: chuẩn hoá, target encoding, chọn đặc trưng, IDF chỉ fit trên phần train của fold.
- OOF (out-of-fold): dự đoán của mỗi fold trên phần val của nó, ghép lại đủ train; dùng để chọn ngưỡng, trọng số ensemble, stacking.

## 3. Tối ưu

- **AdamW**: lr 1e-3 đến 3e-3 khi train từ đầu, 1e-4 đến 3e-4 khi fine-tune backbone pretrained, weight_decay 1e-4 đến 5e-2. CNN train từ đầu đôi khi tốt hơn với **SGD momentum 0,9 + Nesterov**, lr 0,05-0,1.
- **Lịch lr**: OneCycle hoặc warmup + cosine. Batch tăng k lần thì lr tăng khoảng k lần (SGD) hoặc căn k lần (Adam).
- Fine-tune backbone pretrained: lr backbone nhỏ hơn head 10 lần; hoặc đóng băng backbone vài epoch đầu.
- Đóng băng **BatchNorm** khi batch nhỏ hoặc batch có cấu trúc đặc biệt (Nordland: không đóng băng BN thì sụp).
- Gradient accumulation khi thiếu VRAM; `clip_grad_norm_` cho RNN / Transformer.
- **EMA trọng số** (`torch.optim.swa_utils.AveragedModel` với `multi_avg_fn=get_ema_multi_avg_fn(0.999)`): thường tăng 1-2% miễn phí.

## 4. Regularization và augmentation

- Label smoothing 0,1; dropout 0,1-0,3; weight decay; stochastic depth với mạng sâu.
- MixUp (alpha 0,2) / CutMix khi overfit; tắt ở vài epoch cuối.
- **Augmentation khớp với biến đổi của test quan trọng hơn augmentation nhiều**: test bị xoay / JPEG / mờ / nhiễu / tối thì train phải có đúng các biến đổi đó với cường độ tương tự.
- Video: cùng một phép biến đổi cho mọi khung của một video. Ngôn ngữ ký hiệu: cẩn thận lật ngang (đổi tay trái phải).
- Âm thanh: trộn nhiễu theo SNR, dịch thời gian, SpecAugment.
- Văn bản: thay từ đồng nghĩa ít hiệu quả khi train từ đầu; bỏ dấu ngẫu nhiên, lỗi gõ ngẫu nhiên nếu test có.

## 5. Dữ liệu ít nhãn

- **Đặc trưng đóng băng + mô hình nông** (logistic, kNN, LightGBM) thường thắng fine-tune khi dưới vài nghìn mẫu. Trích một lần, cache `.npy`, thử hàng chục head trong vài phút.
- **Pseudo-label** dữ liệu không nhãn được cấp: ngưỡng tin cậy 0,85-0,95, giới hạn số mẫu mỗi lớp để không lệch lớp; lặp 1-2 vòng. Không pseudo-label test nếu luật không cho.
- Nhãn nhiễu: label smoothing, small-loss trick (bỏ các mẫu loss lớn nhất sau vài epoch), co-teaching.

## 6. Suy luận và kết hợp

- **TTA**: lật, xoay (nhóm D4 cho ảnh vệ tinh), nhiều crop, nhiều đoạn video; trung bình **xác suất** (hoặc logit), không trung bình nhãn.
- **Ensemble**: 2-5 mô hình khác seed / khác kiến trúc / khác loại đặc trưng. Khác loại (CNN + đặc trưng thủ công + GBM) lợi hơn cùng loại khác seed.
- Trọng số ensemble và ngưỡng chọn trên OOF, không chọn trên test public (dễ overfit LB khi test public nhỏ).
- Seq2seq: beam search + length penalty, **trung bình trọng số các checkpoint cuối**.

## 7. Tốc độ (để kịp 10-15 lượt nộp)

- Đo trước: in thời gian mỗi batch và ETA sau 20 batch. Đặt ngân sách cho mỗi lần train (ví dụ 15 phút) và tính số epoch theo ngân sách.
- GPU phải bận trên 80% (`nvidia-smi`). Nếu không: CPU đang là nút thắt (đọc ảnh, augmentation).
  - Nạp hết dữ liệu vào RAM / GPU (`uint8`), chia batch bằng `torch.randperm`, augmentation bằng phép toán tensor trên GPU.
  - Ảnh JPEG nhiều: đọc bytes một lần, `torchvision.io.decode_jpeg(list_bytes, device="cuda")`.
  - Video: giải mã một lần, lưu khung đã thu nhỏ ra `.npy`.
- Ảnh nhỏ hơn (128 thay vì 224) cho các vòng thử; độ phân giải đầy đủ cho bản cuối.
- Phần nặng chạy một lần rồi lưu (đặc trưng, logits OOF, logits test); mỗi lượt nộp là một biến thể rẻ.
- Máy thi (GPU đời mới): `torch.autocast("cuda", dtype=torch.bfloat16)`, `torch.backends.cudnn.benchmark = True` cho vòng thử (tắt khi chạy bản cuối cần tái lập).

## 8. Tái lập

```python
def seed_all(s=42):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
```

- `DataLoader(..., generator=torch.Generator().manual_seed(s), worker_init_fn=...)` nếu dùng nhiều worker.
- Một số phép trên GPU không tất định (`index_add`, `scatter_add`, một số conv backward); chấp nhận lệch rất nhỏ, nhưng `main.py` **chỉ suy luận** từ weights đã lưu nên luôn ra cùng kết quả.

## 9. Đọc thêm

- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/): quy trình từ overfit một batch đến tinh chỉnh.
- Leslie Smith, *Super-Convergence* ([arXiv:1708.07120](https://arxiv.org/abs/1708.07120)): OneCycle.
- Zhang và cộng sự, *mixup* ([arXiv:1710.09412](https://arxiv.org/abs/1710.09412)); Yun và cộng sự, *CutMix* ([arXiv:1905.04899](https://arxiv.org/abs/1905.04899)).
- He và cộng sự, *Bag of Tricks for Image Classification* ([arXiv:1812.01187](https://arxiv.org/abs/1812.01187)): label smoothing, warmup, zero-init BN.
