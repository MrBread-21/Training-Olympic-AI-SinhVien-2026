# Lời giải SpeechCommands_KWS - Nhận dạng từ khoá trong âm thanh

Đề: [selfhost_kaggle/SpeechCommands_KWS](../../selfhost_kaggle/SpeechCommands_KWS/). Code: [solve.py](solve.py). Không dùng model pretrained. Cùng khung với bài luyện [d01_am_thanh](../../luyen_tap/CV/d01_am_thanh/).

## Kết quả (chấm bằng đúng metric của cuộc thi, `selfhost_kaggle/tools/cham.py`)

| | Public | Private |
|---|---|---|
| Baseline của đề (log-mel thống kê + logistic) | 0,377 | 0,346 |
| **Lời giải này** | **0,932** | **0,905** |

Val (20% người nói của train, làm nhiễu giống test): 0,836. Thời gian: 14,5 phút trên GTX 1650 lần đầu (gồm đọc 16.000 file wav từ ổ chậm rồi lưu `cache_*.npy`), các lần sau khoảng 8 phút.

## Cách làm

1. Đọc wav một lần, đệm / cắt đủ 16.000 mẫu, lưu cache `.npy` (float16), nạp hết lên GPU.
2. **Log-mel tự tính**: `torch.stft` (n_fft 512, cửa sổ Hann 400, bước 160), 64 bộ lọc mel tam giác, log.
3. **Augmentation trên GPU mỗi batch**, mô phỏng test:
   - dịch thời gian ±0,1 s;
   - 80% đoạn trộn tạp âm theo SNR ngẫu nhiên -2 đến 20 dB: nhiễu trắng, nhiễu hồng, hoặc tổng 3 đoạn train khác (giống tiếng người nền);
   - đổi âm lượng;
   - SpecAugment (2 lần che dải tần và đoạn thời gian).
4. CNN 2D 4 khối (32-64-128-256 kênh, BatchNorm ở đầu vào), AdamW + OneCycle, label smoothing 0,1, **trọng số lớp** căn bậc hai nghịch đảo tần suất (lớp `unknown` chiếm một nửa dữ liệu, metric là macro-F1).
5. Val theo **người nói**; bản cuối train toàn bộ train, 30 epoch.

## Ghi chú

- Điều quyết định là **trộn tạp âm khi train**: ở bài luyện d01, cùng khung mà không trộn nhiễu chỉ được 0,35 trên test có nhiễu.
- Val (0,84) thấp hơn test public (0,93) vì val dùng SNR thấp hơn và ít người nói hơn; xu hướng giữa các biến thể vẫn khớp.
- Hướng tăng tiếp: TTA dịch thời gian, ensemble 3 seed, ResNet-18 (`weights=None`, 1 kênh) trên spectrogram.

## Chạy lại

```
python solve.py <thư_mục upload của đề>     # giải nén private_test.zip vào thư mục làm việc trước nếu muốn dự đoán cả private
python ../../selfhost_kaggle/tools/cham.py SpeechCommands_KWS submission.csv
```
