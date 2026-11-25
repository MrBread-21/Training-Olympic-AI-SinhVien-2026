# Dạng 5 - Video: phân loại hành động, ngôn ngữ ký hiệu, định vị sự kiện

## 1. Nhận dạng đề

- **Phân loại video**: mỗi video một nhãn (vòng khu vực 2025: nhận diện ngôn ngữ ký hiệu tiếng Việt, macro-F1; UCF101_ActionRecognition tự host).
- **Định vị theo thời gian**: nhãn + đoạn `[start_frame, end_frame]` (MSAD: tIoU nếu đúng nhãn).
- **Điểm từng khung**: điểm bất thường mỗi khung (CCTV, xem [dạng 3](03_bat_thuong.md)).

## 2. Metric

Macro-F1 / accuracy cho phân loại; tIoU cho định vị; AUROC cho điểm khung.

## 3. Lộ trình

| Bước | Việc | Ghi chú |
|---|---|---|
| 0 | Baseline BTC, nộp | |
| 1 | **Giải mã một lần**: lấy T = 8-16 khung rải đều mỗi video, resize 112-224, lưu `uint8` `.npy` | đọc video chậm nhất trong cả quy trình |
| 2 | Đặc trưng ImageNet đóng băng **từng khung** (ResNet-50 / ConvNeXt), cache (N, T, D) | 1 lần, vài phút |
| 3 | Gộp thời gian: trung bình + độ lệch chuẩn + hiệu khung (t, t+k) rồi logistic / LightGBM; hoặc GRU / Transformer nhỏ trên chuỗi đặc trưng | thử hàng chục head trong vài phút |
| 4 | Fine-tune CNN 2D theo khung (TSN) hoặc 3D (`r3d_18(weights=None)` nếu không có pretrained video) khi còn thời gian | nặng, cần nhiều dữ liệu |
| 5 | TTA nhiều đoạn (multi-clip), ensemble | |

## 4. Khung code

Lấy khung rải đều:

```python
import cv2
def lay_khung(path, T=16, S=160):
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(n - 1, 0), T).astype(int).tolist()); out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i in idx: out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (S, S)))
        i += 1
    cap.release()
    while len(out) < T: out.append(out[-1] if out else np.zeros((S, S, 3), np.uint8))   # video hỏng / ngắn
    return np.stack(out[:T])                                                         # (T, S, S, 3) uint8
```

`CAP_PROP_FRAME_COUNT` có thể sai với một số định dạng: đọc tuần tự như trên (không `set(CAP_PROP_POS_FRAMES)`) cho chắc, hoặc đếm lại số khung.

Gộp đặc trưng theo thời gian:

```python
def gop(F_):                                   # F_: (N, T, D) đặc trưng từng khung
    T = F_.shape[1]
    d1 = F_[:, 1:] - F_[:, :-1]                # chuyển động mức đặc trưng
    return np.concatenate([F_.mean(1), F_.std(1), np.abs(d1).mean(1), F_[:, :T // 2].mean(1) - F_[:, T // 2:].mean(1)], 1)
```

Mô hình chuỗi nhỏ:

```python
class GRUHead(nn.Module):
    def __init__(self, D, C, h=256):
        super().__init__(); self.gru = nn.GRU(D, h, batch_first=True, bidirectional=True); self.fc = nn.Linear(2 * h, C)
    def forward(self, x):                       # (B, T, D)
        o, _ = self.gru(x); return self.fc(o.mean(1))
```

Định vị đoạn (MSAD): điểm từng khung từ một bộ phân loại khung, làm mượt, lấy đoạn liên tục lớn nhất vượt ngưỡng (thuật toán Kadane trên `điểm - ngưỡng`). Khi không chắc, trả "đoạn kỳ vọng" tối đa tIoU trung bình theo phân bố vị trí sự kiện trong train.

## 5. Nút vặn

1. Số khung T và độ phân giải (cân bằng thời gian giải mã).
2. Augmentation **giống nhau cho mọi khung** của một video (cùng crop, lật, màu).
3. Kênh chuyển động: hiệu hai khung liên tiếp làm đầu vào phụ.
4. Ngôn ngữ ký hiệu: lật ngang đổi tay, chỉ dùng nếu đề không phân biệt tay; tập trung vùng tay và mặt (crop trung tâm phía trên).
5. Multi-clip TTA: nhiều cách lấy T khung (lệch pha), trung bình xác suất.
6. Validation chia theo **người / cảnh / video gốc** (UCF101: các clip cắt từ cùng video gốc rất giống nhau).

## 6. Bẫy

- Clip cắt từ cùng một video gốc nằm cả ở train và val: val ảo cao.
- fps và độ dài khác nhau giữa video: luôn lấy mẫu theo tỉ lệ (linspace), không theo số khung cố định từ đầu.
- Giải mã video trên Windows trong DataLoader nhiều worker hay treo: giải mã trước thành `.npy`.

## 7. Bài luyện

- Code tay: [d26_video](../../luyen_tap/CV/d26_video/) (khung rải đều, cache, gộp thời gian, tIoU kỳ vọng).
- Thi thử: [UCF101_ActionRecognition](../../selfhost_kaggle/UCF101_ActionRecognition/) (baseline 0,864).
- Lời giải tham khảo: [challenges/MSAD](../../challenges/MSAD/) (đặc trưng chuyển động + màu + GBM, định vị đoạn; bản không pretrained 0,415 public).

## 8. Đọc thêm

- Wang và cộng sự, *Temporal Segment Networks* ([arXiv:1608.00859](https://arxiv.org/abs/1608.00859)).
- Tran và cộng sự, *A Closer Look at Spatiotemporal Convolutions* (R(2+1)D, r3d, [arXiv:1711.11248](https://arxiv.org/abs/1711.11248)).
