# Bài luyện tập VIQA · Hỏi đáp trích xuất tiếng Việt

*Bài ôn luyện của đội tuyển PHENIKAA AI Hackathon 2026. Dữ liệu: SQuAD, XQuAD, MLQA (CC BY-SA).*

## 1. Bối cảnh
Cho một đoạn văn Wikipedia tiếng Việt và một câu hỏi, hãy tìm **đoạn văn bản trả lời** (một khoảng liên tục trong đoạn văn). Dữ liệu huấn luyện tiếng Việt rất ít; phần lớn là tiếng Anh (SQuAD), nên cần mô hình đa ngữ hoặc chuyển giao ngôn ngữ (zero-shot cross-lingual), tương tự cuộc thi chaii trên Kaggle.

## 2. Nhiệm vụ
Với mỗi mẫu trong `test.json` (`id`, `context`, `question`), trả về chuỗi `answer` copy từ `context`.

## 3. Dữ liệu (gói `VIQA_data.zip`)
| File | Nội dung |
|---|---|
| `train_en.json` | 87 599 mẫu SQuAD v1.1 (tiếng Anh) có `answers[{text, answer_start}]` |
| `train_vi.json` | 1 190 mẫu tiếng Việt (XQuAD-vi) |
| `validation_vi.json` | 511 mẫu tiếng Việt (MLQA-vi, dev) để tự đánh giá |
| `test.json` | 2 000 mẫu tiếng Việt (lấy mẫu từ MLQA-vi test), **không có đáp án** |
| `sample_submission.csv`, `starter.py`, `scorer.py` | file nộp mẫu, baseline từ vựng, hàm chấm (Jaccard, EM) |

**Giấy phép.** SQuAD v1.1 (Rajpurkar et al.): **CC BY-SA 4.0**. XQuAD (Artetxe et al., ACL 2020): **CC BY-SA 4.0**. MLQA (Lewis et al., 2020): **CC BY-SA 3.0**. Gói này là bản lấy mẫu/định dạng lại; dữ liệu phát lại theo giấy phép **CC BY-SA** (chia sẻ tương tự, ghi nguồn).

## 4. File nộp
`submission.csv`, UTF-8, dòng đầu `id,answer`, **đúng một dòng cho mỗi mẫu test** (đặt trong dấu nháy kép nếu có dấu phẩy).

## 5. Chấm điểm
Chuẩn hóa chữ thường, bỏ dấu câu, tách theo khoảng trắng (âm tiết). **Điểm chính: Jaccard mức từ** trung bình (như chaii), điểm phụ: Exact Match. Public/private chia theo đoạn văn (40/60).

## 6. Luật
Được dùng mô hình tiền huấn luyện có giấy phép cho phép (ghi nguồn). **Không** huấn luyện hay tra cứu trên MLQA/XQuAD/UIT-ViQuAD ngoài các file được cung cấp, không dùng đáp án test công khai trên mạng. Tối đa 20 lần nộp mỗi ngày.

## 7. Gợi ý
Tinh chỉnh mô hình đa ngữ (XLM-R, mBERT, PhoBERT cho tiếng Việt nếu đã tinh chỉnh span) trên SQuAD rồi trên `train_vi`; dịch dữ liệu; hậu xử lý span (độ dài, dấu câu), ensemble.

## 8. Nguồn cảm hứng
Kaggle chaii (Hindi/Tamil QA); IOAI 2026 (CC BY 4.0) về bài toán đáp trả theo ngữ cảnh.
