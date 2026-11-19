# Bài luyện tập CAPTION · Mô tả ảnh (image captioning)

*Bài ôn luyện của đội tuyển PHENIKAA AI Hackathon 2026. Dữ liệu: COCO captions (huấn luyện) và nocaps (kiểm tra).*

## 1. Bối cảnh
Viết một câu mô tả bằng tiếng Anh cho mỗi ảnh. Tập huấn luyện từ COCO (80 loại vật thể quen thuộc); tập kiểm tra từ **nocaps**: ảnh Open Images có nhiều vật thể và cảnh **chưa từng xuất hiện** trong COCO, nên mô hình cần khái quát hóa (novel object captioning).

## 2. Nhiệm vụ
Với mỗi ảnh test, sinh **một câu mô tả** ngắn gọn, tự nhiên.

## 3. Dữ liệu (gói `CAPTION_data.zip`)
| Thư mục / file | Nội dung |
|---|---|
| `train/` + `train_captions.json` | 3 500 ảnh COCO (cạnh dài 384 px) và 5 mô tả/ảnh |
| `val/` + `val_captions.json` | 300 ảnh COCO có mô tả để tự đánh giá (cùng miền với train, dễ hơn test) |
| `test/` | 800 ảnh nocaps, **mô tả bị ẩn** (mỗi ảnh có 10 mô tả tham chiếu trong đáp án) |
| `sample_submission.csv`, `starter.py`, `scorer.py` | file nộp mẫu, baseline truy hồi, hàm chấm (CIDEr-D, BLEU-4, ROUGE-L) |

**Giấy phép.** Ảnh test: nocaps (Agrawal et al., ICCV 2019), ảnh từ Open Images giấy phép **CC BY 2.0**, mô tả do tác giả nocaps biên soạn. Mô tả COCO: **CC BY 4.0**; ảnh COCO theo điều khoản của Flickr (từng ảnh có giấy phép Creative Commons riêng). Chỉ dùng nội bộ cho nghiên cứu/học tập; ghi nguồn khi công bố kết quả. Ảnh trong gói đã được thu nhỏ.

## 4. File nộp
`submission.csv`, UTF-8, dòng đầu `id,caption`, **đúng một dòng cho mỗi ảnh test**. Nếu câu có dấu phẩy, đặt trong dấu nháy kép (thư viện `csv` tự làm).

## 5. Chấm điểm
**Điểm chính: CIDEr-D** (thang ~0–1+, nhân 10 như bản chuẩn), điểm phụ: BLEU-4; ROUGE-L có trong chi tiết. Public 40% / private 60% theo ảnh. Mã chấm thuần Python ở `scorer.py` (tách từ bằng chữ thường và số).

## 6. Luật
Được dùng mô hình tiền huấn luyện có giấy phép cho phép (ghi nguồn). Không tra cứu mô tả nocaps từ bản gốc, không viết mô tả tay cho ảnh test. Tối đa 20 lần nộp mỗi ngày.

## 7. Gợi ý
Mô hình vision-language (BLIP, GIT, ClipCap...), tinh chỉnh trên train, giải mã beam search, độ dài câu vừa phải, prompt cho mô hình đa phương thức nhỏ nếu giấy phép cho phép.

## 8. Nguồn cảm hứng
Kaggle: các bài image captioning (Flickr8k/COCO); nocaps; phong cách bài của IOAI 2026 (CC BY 4.0).
