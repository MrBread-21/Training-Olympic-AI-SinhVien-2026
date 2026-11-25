# 01 - Metric: công thức, cách tối ưu, bẫy

Code của mọi metric dưới đây nằm trong [metric_tu_viet.py](metric_tu_viet.py) (chỉ cần numpy, đã đối chiếu với sklearn và nltk: `python metric_tu_viet.py`). Phòng thi không có mạng nên `evaluate.load(...)` không chạy được, phải tự viết.

Nguyên tắc chung: **đọc metric trước khi chọn mô hình**. Metric quyết định nên dự đoán nhãn hay xác suất, có nên chỉnh ngưỡng, có nên để trống, và lỗi nào bị phạt nặng.

## Phân loại

### Accuracy và macro-F1

- `F1_c = 2TP / (2TP + FP + FN)` cho từng lớp c; **macro-F1** = trung bình không trọng số các lớp.
- Macro-F1 coi lớp hiếm nặng bằng lớp đông. Đoán toàn lớp đông chỉ được khoảng `1 / số_lớp`.
- Tối ưu: trọng số lớp trong loss, oversampling, và **chỉnh ngưỡng / hệ số từng lớp trên val**. Cách nhanh: nhân xác suất lớp c với hệ số `w_c`, tìm `w_c` theo vòng lặp toạ độ trên OOF.
- Bẫy: lớp có trong test nhưng không có trong val thì val không phản ánh được; lớp chỉ có vài mẫu val làm macro-F1 dao động mạnh.

### AUROC

- Xác suất một mẫu dương ngẫu nhiên có điểm cao hơn một mẫu âm ngẫu nhiên. Chỉ phụ thuộc **thứ tự**, không cần hiệu chỉnh xác suất.
- Nộp hằng số được 0,5; nộp điểm đảo chiều được `1 - AUROC`. Kiểm tra chiều: nếu val dưới 0,5 rõ rệt thì đổi dấu.
- Khi ensemble nhiều mô hình có thang điểm khác nhau: **trung bình hạng** (rank average) an toàn hơn trung bình điểm.
- Bẫy (bài CCTV): AUROC gộp mọi khung của mọi video, nên chênh lệch giữa các video chiếm phần lớn tín hiệu; chuẩn hoá điểm theo từng video có thể làm hỏng.

### Average Precision (AP), mAP

- AP = trung bình precision tại vị trí của mỗi mẫu dương khi xếp điểm giảm dần. Nhạy với các mẫu điểm cao nhất hơn AUROC; dùng khi dương rất hiếm.
- mAP (đa nhãn / phát hiện vật thể) = trung bình AP của các lớp.

## Hồi quy

- **RMSE** phạt nặng sai số lớn: dự đoán co về trung bình là an toàn. Mô hình hay "co về giữa" (DimABSA: hệ số góc 0,73), có thể giãn nhẹ dự đoán quanh trung bình nếu val cho thấy lợi.
- **RMSE_VA** (DimABSA) = `sqrt(1/N tổng[(Vp - Vg)^2 + (Ap - Ag)^2])`, chia cho N cặp chứ không phải 2N. Luôn **kẹp dự đoán vào [1, 9]** và làm tròn 2 chữ số: giá trị ngoài miền bị coi là không hợp lệ.
- Đoán trung bình train là baseline: RMSE_VA khoảng 1,8-2,2 với dữ liệu nhà hàng.

## Phân đoạn

- **Dice** `= 2|A ∩ B| / (|A| + |B|)`, **IoU** `= |A ∩ B| / |A ∪ B|`; `Dice = 2 IoU / (1 + IoU)`.
- Thường tính trên **từng ảnh rồi lấy trung bình**: ảnh có vật nhỏ nặng bằng ảnh có vật lớn. Ảnh không có vật: cả hai rỗng thường được 1, đoán ra vài điểm ảnh thì 0. Bỏ các vùng dự đoán quá nhỏ (dưới vài chục điểm ảnh).
- Loss nên là `BCE + Dice` (Dice loss trực tiếp tối ưu metric). Ngưỡng nhị phân hoá chọn trên val (thường 0,4-0,5).
- File nộp RLE: kiểm tra thứ tự duyệt (theo hàng hay theo cột), chỉ số bắt đầu từ 0 hay 1, rồi giải mã lại và so với mask trước khi nộp.

## Truy xuất

- **Recall@K**: tỉ lệ truy vấn có ít nhất một kết quả đúng trong K kết quả đầu. **MRR@K**: trung bình `1 / hạng của kết quả đúng đầu tiên`.
- Nordland: "đúng" là lệch tối đa 10 khung, nên nhiều ảnh database cùng đúng. Đề chỉ nhận **một** dự đoán cho mỗi truy vấn: Recall@1.
- Tối ưu: re-ranking (kiểm tra hình học, lấy trung bình với láng giềng), query expansion, dùng tính liên tục thời gian nếu dữ liệu cho phép.

## So khớp văn bản

- **Exact Match**: sau chuẩn hoá (chữ thường, bỏ dấu câu, gộp khoảng trắng) phải trùng hoàn toàn.
- **F1 theo từ** (SQuAD): đếm từ trùng có lặp; `P = trùng / số từ dự đoán`, `R = trùng / số từ đáp án`. Ví dụ đáp án "Lâm Bá Kiệt", dự đoán "tên Lâm Bá Kiệt": F1 = 0,857.
- **Jaccard theo âm tiết**: `|A ∩ B| / |A ∪ B|` trên tập từ (không đếm lặp).
- SQuAD 2.0: câu không có đáp án, để trống được 1, trả lời gì cũng 0. Nếu 30% câu hỏi không có đáp án thì nộp toàn rỗng đã được khoảng 0,3: **ngưỡng để trống là nút vặn lớn nhất**.
- Bẫy: Unicode NFC và NFD (chữ "ệ" có thể là 1 hoặc 3 code point). Luôn `unicodedata.normalize("NFC", s)` cả dự đoán lẫn dữ liệu.

## Sinh chuỗi

### BLEU-4

- Trung bình nhân của precision n-gram 1-4 (có cắt số lần đếm theo tham chiếu), nhân hệ số phạt câu ngắn `BP = exp(1 - r/c)` khi tổng độ dài dự đoán c ngắn hơn tham chiếu r. Tính **mức corpus** (cộng dồn đếm của mọi câu rồi mới chia), không phải trung bình BLEU từng câu.
- SacreBLEU tự tách từ theo quy tắc riêng; nếu đề dùng SacreBLEU thì đừng tự tách khác đi. Ra 0 khi không có 4-gram nào trùng.
- Tối ưu: beam search 4-5 với length penalty; độ dài đầu ra hợp lý (quá ngắn bị BP phạt, quá dài giảm precision).

### chrF

F-beta (beta = 2, ưu tiên recall) trên n-gram **ký tự** 1-6. Bền với lỗi hình thái và dấu tiếng Việt hơn BLEU.

### CIDEr-D (caption)

- Mỗi câu thành vector TF-IDF của n-gram 1-4; IDF tính trên **các câu tham chiếu của tập đang chấm**. Cosine với từng tham chiếu, có cắt (lặp từ không được thêm điểm) và phạt chênh độ dài `exp(-Δ^2 / (2 x 6^2))`; trung bình 4 bậc và các tham chiếu, nhân 10.
- N-gram xuất hiện ở mọi ảnh ("trong bức ảnh") gần như không có điểm; n-gram đặc trưng (tên vật thể, hành động) có điểm cao.
- Tối ưu: chọn câu **đồng thuận** trong các ứng viên (MBR theo CIDEr), độ dài câu gần độ dài trung bình của tham chiếu.

## Gán nhãn chuỗi

- **F1 mức thực thể** (seqeval): một thực thể đúng khi trùng loại, vị trí đầu và cuối. Sai một token ở biên là sai cả thực thể.
- Sửa chuỗi BIO không hợp lệ sau khi dự đoán (I- đứng sau O thì đổi thành B-) hoặc dùng CRF / giải mã Viterbi với ràng buộc chuyển nhãn.

## Video

- **tIoU** của hai đoạn `[đầu, cuối]` = độ dài giao / độ dài hợp. Đoạn dự đoán quá rộng hay quá hẹp đều mất điểm.
- MSAD: sai nhãn thì 0 dù đoạn đúng, nên khi không chắc nhãn, đoạn dự đoán nên là "đoạn kỳ vọng" tối đa tIoU trung bình theo phân bố.

## Cấu trúc (DimABSA cF1)

- Một bộ tứ dự đoán khớp khi (aspect, category, opinion) trùng hoàn toàn sau chuyển chữ thường; khi khớp, `cTP = max(0, 1 - khoảng cách VA / sqrt(128))`.
- Hai dự đoán trùng cùng khớp một nhãn chuẩn: nhãn đó nhận 0 và cả hai tính là dự đoán sai. **Khử trùng lặp trước khi nộp**.
- Lệch ranh giới cụm từ mất trọn điểm, sai VA chỉ bị trừ một phần: ưu tiên trích đúng chuỗi.

## Điểm quy đổi của OlpAI

`điểm bài = (điểm - Min) / (Max - Min)`, Min là baseline BTC, Max là mô hình mạnh của BTC. Với metric thấp-là-tốt (RMSE) thì đổi chiều. Hệ quả:

- Một bài chưa vượt baseline được 0 (hoặc âm). Đưa **mọi bài** lên trên Min trước khi tối ưu sâu một bài.
- Tăng 0,01 macro-F1 ở bài có `Max - Min = 0,05` đáng giá bằng tăng 0,1 ở bài có `Max - Min = 0,5`. Chia thời gian theo **điểm quy đổi tăng thêm mỗi giờ**.
