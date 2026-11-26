# Dạng 14 - Cảm xúc đa chiều theo khía cạnh (DimABSA, đề NLP OlpAI'26)

## 1. Nhận dạng đề

Đề NLP OlpAI'26 (tài liệu AIO2026, repo Purin1410/Olympic_AI_MET_2026_preliminary_round) gồm hai bài con độc lập, dữ liệu JSON Lines:

- **Subtask 1** (hồi quy): cho câu và danh sách khía cạnh, dự đoán (valence, arousal) trong [1, 9] cho từng khía cạnh, làm tròn 2 chữ số. Chấm RMSE_VA.
- **Subtask 2** (trích cấu trúc): chỉ cho câu, tìm tập bộ tứ (aspect, category, opinion, valence, arousal). Aspect / opinion chép nguyên văn từ câu, ẩn thì `"NULL"`. Chấm cF1.
- Đề thật: tiếng Việt có teencode, lỗi gõ, chêm tiếng Anh; 10 category (`APP#PERFORMANCE`, `APP#UI_UX`, `STAFF#ATTITUDE`, `STAFF#PROFESSIONALISM`, `PRODUCT#QUALITY`, `PRODUCT#DESIGN`, `SERVICE#TIME`, `SERVICE#GENERAL`, `PRICE#RATES`, `PRICE#PROMO`). Luật vòng khu vực: không pretrained NLP.

```
{"id": "PUB_001", "text": "App lag quá, nhưng shipper thì dễ thương xỉu.", "aspects": [{"term": "App"}, {"term": "shipper"}]}
{"id": "PUB_001", "aspects": [{"aspect": "App", "valence": 2.00, "arousal": 7.50}, {"aspect": "shipper", "valence": 8.00, "arousal": 6.00}]}
```

## 2. Metric

- RMSE_VA = `sqrt(1/N tổng[(Vp - Vg)^2 + (Ap - Ag)^2])`. Kẹp [1, 9], làm tròn 2 chữ số.
- cF1: khớp khi (aspect, category, opinion) trùng hoàn toàn sau chữ thường; `cTP = max(0, 1 - khoảng cách VA / sqrt(128))`; dự đoán trùng lặp bị phạt. Lệch ranh giới cụm mất trọn điểm, sai VA chỉ bị trừ một phần.

## 3. Đặc điểm dữ liệu cần khai thác (từ phân tích bản tiếng Anh)

- Valence lệch về tích cực (đỉnh 7-8), arousal hẹp (độ lệch chuẩn 1,1) và **tăng khi valence xa 5** (hình chữ V, Spearman 0,94): mô hình có thể học arousal qua valence.
- Câu trung bình 1,6 bộ tứ ở train, 2,1 ở test; 61% câu train chỉ có 1 bộ.
- Category lệch nặng (3 nhãn chiếm 69%).
- NULL: bản tiếng Anh có NULL ở train nhưng không ở test (bỏ NULL khi nộp được lợi); đề tiếng Việt cho phép NULL ở test, phải giữ.

## 4. Lộ trình Subtask 1 (hồi quy VA, không pretrained)

| Bước | Việc | Kỳ vọng (bản tiếng Anh) |
|---|---|---|
| 0 | Đoán trung bình train | RMSE_VA khoảng 1,84 (dev) / 2,2 (test) |
| 1 | **Cửa sổ quanh khía cạnh** (±6 từ) + TF-IDF từ và ký tự + Ridge riêng V và A | 1,47 (dev) / 1,90 (test public) |
| 2 | Từ điển cảm xúc học từ train: trung bình V, A của mỗi opinion / từ, khoảng cách tới khía cạnh; LightGBM trên đặc trưng | |
| 3 | BiGRU từ đầu trên câu, **đánh dấu vị trí khía cạnh** (embedding vị trí tương đối hoặc token đặc biệt quanh khía cạnh), đầu ra 2 giá trị | |
| 4 | Loss MSE có trọng số học theo độ bất định (LogSigma), ensemble 3 seed, kẹp và làm tròn | tài liệu AIO: RoBERTa-large 1,10 (có pretrained) |

```python
class VAReg(nn.Module):
    def __init__(self, V, d=200, h=192):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0); self.dau = nn.Embedding(3, 16)    # 0 pad / ngoài, 1 khía cạnh, 2 câu
        self.gru = nn.GRU(d + 16, h, batch_first=True, bidirectional=True); self.fc = nn.Linear(4 * h, 2)
        self.log_var = nn.Parameter(torch.zeros(2))                                    # LogSigma: lr riêng lớn (5e-2)
    def forward(self, x, m):                       # x: (B, L) id từ; m: (B, L) 1 ở token khía cạnh, 2 ở token khác, 0 pad
        o, _ = self.gru(torch.cat([self.emb(x), self.dau(m)], -1))
        asp = (o * (m == 1).unsqueeze(-1)).sum(1) / (m == 1).sum(1, keepdim=True).clamp(min=1)
        mx = o.masked_fill((m == 0).unsqueeze(-1), -1e4).max(1).values
        return 5 + 4 * torch.tanh(self.fc(torch.cat([asp, mx], 1)))                    # nằm sẵn trong (1, 9)

def loss_logsigma(pred, y, log_var):              # Kendall và cộng sự 2018
    mse = ((pred - y) ** 2).mean(0)                 # (2,) MSE của V và A
    return (0.5 * torch.exp(-log_var) * mse + 0.5 * log_var).sum()
# optimizer: hai nhóm tham số, log_var dùng lr 5e-2 (nếu chung lr 2e-5 thì sigma đứng yên, tài liệu AIO đã chỉ ra)
```

## 5. Lộ trình Subtask 2 (trích bộ tứ, không pretrained)

Tách thành các bước con, mỗi bước một mô hình nhỏ:

1. **Gán nhãn BIO** cho aspect và opinion trên token (`B-ASP`, `I-ASP`, `B-OPI`, `I-OPI`, `O`): BiLSTM + ký tự + Viterbi ([dạng 10](10_gan_nhan_chuoi.md)).
2. **Ghép cặp** (aspect, opinion): với mọi cặp ứng viên trong câu, bộ phân loại nhị phân (đặc trưng: khoảng cách, thứ tự, có dấu phẩy / "nhưng" ở giữa, embedding hai cụm); hoặc ghép mỗi opinion với aspect gần nhất.
3. **Category** cho mỗi cặp: phân loại (TF-IDF của aspect + opinion + cửa sổ, hoặc từ điển aspect thành category học từ train).
4. **VA** cho mỗi cặp: mô hình Subtask 1 (đầu vào: câu + khía cạnh + opinion).
5. NULL: aspect ẩn (opinion có mà không có aspect gần) gán `NULL` với category hay gặp nhất ("RESTAURANT#GENERAL" / tương đương); chỉ giữ nếu đề cho phép NULL ở test.
6. **Khử trùng lặp** bộ (aspect, category, opinion) trước khi nộp.

Baseline từ điển (bài tự host): khía cạnh và opinion đã gặp trong train, ghép gần nhất, VA trung bình của opinion: cF1 0,41 dev / 0,305 test.

## 6. Nút vặn

1. Subtask 2: **ranh giới cụm** quan trọng nhất (cF1 mất khoảng 0,30 ở trích xuất, chỉ 0,05 ở VA theo phân tích của AIO). Hậu xử lý cắt / nối cụm theo mẫu hay gặp ở train ("rất ngon", "không ngon": giữ phủ định và trạng từ trong opinion nếu train làm vậy).
2. Self-consistency kiểu nghèo: ensemble 5 seed của bước BIO, giữ cụm được đa số chọn.
3. Subtask 1: tận dụng quan hệ V-A (dự đoán A từ |V - 5|), co về trung bình khi không chắc.
4. Tiếng Việt: chuẩn hoá teencode ("ko", "k" thành "không", "dc" thành "được"), NFC, giữ emoji như token (mang cảm xúc), **không** bỏ dấu.
5. Kẹp [1, 9], làm tròn 2 chữ số, đúng tên trường JSON.

## 7. Bẫy

- Ghép dự đoán với khía cạnh theo **chuỗi**: chép nguyên `term` của đầu vào, không chuẩn hoá lại.
- Trùng lặp bộ tứ bị phạt nặng trong cF1.
- Câu train đã tách từ / viết thường còn test là câu gốc (bản tiếng Anh): chuẩn hoá cả hai giống nhau, nhưng **xuất cụm theo chuỗi gốc của câu test**.
- Thiếu dòng / thiếu khía cạnh: bộ chấm gốc báo lỗi cả file.

## 8. Bài luyện

- Code tay: [d04_dimabsa](../../luyen_tap/CV/d04_dimabsa/) (dữ liệu tiếng Việt giả lập: VA cho khía cạnh, BIO + ghép cặp + category).
- Thi thử: [DimABSA_VA](../../selfhost_kaggle/DimABSA_VA/) (baseline 1,901), [DimABSA_Quad](../../selfhost_kaggle/DimABSA_Quad/) (baseline 0,305).
- Tài liệu đề: `OAI2-Dimensional Aspect-Based Sentiment Analysis.pdf` (AIO2026), repo Purin1410/Olympic_AI_MET_2026_preliminary_round (LogSigma, SCSG, phân tích lỗi).

## 9. Đọc thêm

- SemEval-2026 Task 3 DimABSA: github.com/DimABSA/DimABSA2026 (dữ liệu, bộ chấm `metrics_subtask_1_2_3.py`).
- Kendall, Gal, Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses* ([arXiv:1705.07115](https://arxiv.org/abs/1705.07115)).
- Russell, *A circumplex model of affect* (1980): mặt phẳng valence-arousal.
- Zhang và cộng sự, *Aspect Sentiment Quad Prediction as Paraphrase Generation* ([arXiv:2110.00796](https://arxiv.org/abs/2110.00796)).
