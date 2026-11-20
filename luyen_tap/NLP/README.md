# BỘ 16 BÀI LUYỆN NLP – OLP AI 2026

## 1. Giới thiệu

Thư mục này tổng hợp 16 bài tập NLP (Xử lý ngôn ngữ tự nhiên) nhằm giúp các thành viên ôn luyện kiến thức, làm quen với nhiều dạng bài toán và chuẩn bị cho kỳ thi OLP AI 2026.

Các bài tập được chia thành 8 nhóm, mỗi nhóm gồm 2 bài. Bộ bài tập kết hợp các cuộc thi NLP trên Kaggle, DrivenData và các bộ dữ liệu nghiên cứu, giúp thành viên rèn luyện kỹ năng xử lý dữ liệu, xây dựng mô hình, đánh giá kết quả và tối ưu điểm số.

## 2. Danh sách bài tập

### 01. Phân loại văn bản (Text Classification)

**Bài 1: Disaster Tweets**
- Mô tả: Phân loại tweet có đề cập đến thảm họa hay không.
- Metric: F1-score.
- Đề và dữ liệu: https://www.kaggle.com/competitions/nlp-getting-started
- Trạng thái: Cuộc thi đang hoạt động, có thể nộp bài.

**Bài 2: Jigsaw Toxic Comments**
- Mô tả: Phân loại bình luận thành 6 loại độc hại, một bình luận có thể thuộc nhiều loại.
- Metric: Mean ROC-AUC.
- Đề và dữ liệu: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- Trạng thái: Cuộc thi đã kết thúc, sử dụng để luyện tập và đánh giá offline.

### 02. Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis – ABSA)

**Bài 1: UIT-ViSFD**
- Mô tả: Phân tích 10 khía cạnh và 3 mức cảm xúc trong bình luận tiếng Việt.
- Metric: Macro-F1.
- Dữ liệu: https://huggingface.co/datasets/visolex/ViSFD
- Trạng thái: Bộ dữ liệu nghiên cứu, thực hành và đánh giá offline.

**Bài 2: UIT-ABSA**
- Mô tả: Phát hiện khía cạnh và xác định cảm xúc tương ứng trong câu.
- Metric: Aspect F1, Sentiment F1.
- Dữ liệu: https://nlp.uit.edu.vn/datasets
- Trạng thái: Bộ dữ liệu nghiên cứu, thực hành và đánh giá offline.

### 03. Chuẩn hóa văn bản (Text Normalization)

**Bài 1: Vietnamese Diacritic Restoration**
- Mô tả: Khôi phục dấu tiếng Việt cho văn bản bị mất dấu.
- Metric: Character Accuracy, Word Accuracy.
- Dữ liệu: https://huggingface.co/datasets/nrl-ai/vn-diacritic-eval
- Trạng thái: Bộ dữ liệu đánh giá, cần kết hợp dữ liệu huấn luyện để thực hành.

**Bài 2: Vietnamese Spell Correction**
- Mô tả: Sửa lỗi chính tả và chuẩn hóa văn bản tiếng Việt.
- Metric: Exact Match, Word Accuracy.
- Dữ liệu: https://huggingface.co/datasets/nrl-ai/vn-spell-correction-eval
- Trạng thái: Bộ dữ liệu đánh giá, cần kết hợp dữ liệu huấn luyện để thực hành.

### 04. Trích xuất và phân tích văn bản (Text Extraction & Authorship Attribution)

**Bài 1: Tweet Sentiment Extraction**
- Mô tả: Trích xuất cụm từ trong tweet thể hiện cảm xúc.
- Metric: Jaccard Score.
- Đề và dữ liệu: https://www.kaggle.com/competitions/tweet-sentiment-extraction
- Trạng thái: Cuộc thi đã kết thúc, có thể thực hành và đánh giá offline.

**Bài 2: Name That LLM**
- Mô tả: Xác định mô hình ngôn ngữ lớn (LLM) đã tạo ra một đoạn văn bản dựa trên đặc trưng ngôn ngữ và phong cách viết.
- Metric: Balanced Accuracy.
- Dữ liệu: https://www.kaggle.com/competitions/name-that-llm
- Trạng thái: Cuộc thi Kaggle đang diễn ra, có train, test, sample_submission và leaderboard. Hạn cuối nộp bài dự kiến 13/01/2027.

### 05. Sinh văn bản và suy luận ngôn ngữ (Text Generation & Natural Language Inference)

**Bài 1: What's Up, Docs? – Document Summarization**
- Mô tả: Xây dựng hệ thống tự động tóm tắt nội dung các bài nghiên cứu khoa học thành một đoạn abstract.
- Metric: ROUGE-2 F1.
- Đề và dữ liệu: https://www.drivendata.org/competitions/297/whats-up-docs/
- Trạng thái: Cuộc thi thực hành trên DrivenData, cho phép đăng ký, tạo dự đoán và nộp submission.

**Bài 2: Contradictory, My Dear Watson**
- Mô tả: Xác định quan hệ ngữ nghĩa giữa hai câu, gồm ba nhãn: Entailment, Contradiction và Neutral. Hỗ trợ nhiều ngôn ngữ.
- Metric: Accuracy.
- Đề và dữ liệu: https://www.kaggle.com/competitions/contradictory-my-dear-watson
- Trạng thái: Cuộc thi Kaggle hoạt động với rolling leaderboard, có thể nộp bài.

### 06. Biểu diễn và truy hồi (Embedding & Retrieval)

**Bài 1: U.S. Patent Phrase Matching**
- Mô tả: Đánh giá mức độ tương đồng ngữ nghĩa giữa hai cụm từ trong lĩnh vực sáng chế.
- Metric: Pearson Correlation.
- Đề và dữ liệu: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching
- Trạng thái: Cuộc thi đã kết thúc, có thể thực hành và đánh giá offline.

**Bài 2: Quora Question Pairs**
- Mô tả: Xác định hai câu hỏi có cùng ý nghĩa hay không.
- Metric: Log Loss.
- Đề và dữ liệu: https://www.kaggle.com/competitions/quora-question-pairs
- Trạng thái: Cuộc thi đã kết thúc, có thể thực hành và đánh giá offline.

### 07. Đánh giá LLM và kiểm chứng thông tin (LLM Evaluation & Fact Verification)

**Bài 1: LLM Classification Finetuning**
- Mô tả: Dự đoán câu trả lời nào trong hai câu trả lời của các mô hình ngôn ngữ lớn được người dùng đánh giá cao hơn dựa trên dữ liệu Chatbot Arena.
- Metric: Log Loss.
- Đề và dữ liệu: https://www.kaggle.com/competitions/llm-classification-finetuning

**Bài 2: ViFactCheck**
- Mô tả: Xây dựng hệ thống kiểm chứng thông tin tiếng Việt dựa trên bằng chứng, xác định mức độ chính xác của các tuyên bố.
- Metric: F1-score (tùy cách thiết lập).
- Nguồn dữ liệu: https://github.com/lienlt10/awesome-vietnamese-datasets

### 08. Bài toán kiểu IOAI (IOAI-style NLP Challenges)

**Bài 1: IOAI 2025 – Hallucination Detection**
- Mô tả: Tìm hiểu bài toán phát hiện nội dung do AI tạo ra nhưng không chính xác hoặc không có căn cứ.
- Metric: Theo yêu cầu của từng bài.
- Nguồn tham khảo: https://github.com/open-cu/awesome-ioai-tasks

**Bài 2: IOAI 2026 – Potato**
- Mô tả: Tìm hiểu và thực hành bài toán Potato trong bộ đề IOAI 2026.
- Metric: Theo yêu cầu của đề.
- Đề và tài liệu: https://github.com/IOAI-official/IOAI-2026
