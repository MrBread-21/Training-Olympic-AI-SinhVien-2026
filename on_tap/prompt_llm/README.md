# Luyện prompt cho LLM phòng thi (chung kết OlpAI'26)

Phòng thi chung kết có một chatbot offline: **DeepSeek-R1-Distill-Qwen-32B**. Mỗi cuộc trò chuyện chỉ được **2.000 token**, tính cả tin nhắn gửi đi, câu trả lời và lịch sử. Thư mục này gồm:

- khung prompt đã thử thật trên chatbot mô phỏng;
- 16 prompt mẫu cho 10 bài luyện trong `luyen_tap/CV`, gồm cả 2 prompt sửa lỗi;
- công cụ gửi prompt và ghép code;
- kết quả chấm code do chatbot viết bằng bộ chấm của từng bài.

```
prompt_llm/
├── README.md            hướng dẫn này
├── dau_file.py          phần đầu file chuẩn: import, seed_all(42), dev, use_bf16, amp()
├── khung_1_phan.txt     khung prompt cho bài viết trong một lần (code dưới khoảng 90 dòng)
├── khung_2_phan.txt     khung chia 2 cuộc trò chuyện, nối bằng CONTRACT
├── khung_sua_loi.txt    khung nhờ sửa một đoạn code lỗi
├── hoi_llm.py           gửi prompt, ghép code thành file chạy được
├── mau/                 prompt mẫu đã thử với các bài luyện
└── ket_qua/             code chatbot viết từ prompt cuối, đã đạt bộ chấm (dòng sửa tay đánh dấu "SỬA TAY")
```

## 1. Hiểu chatbot trước khi viết prompt

Đo trên trang mô phỏng ngày 28/11:

| Đặc điểm | Hệ quả khi viết prompt |
|---|---|
| 2.000 token cho **cả** cuộc trò chuyện. Hết token thì câu trả lời bị cắt giữa chừng. | Prompt khoảng 2.500-3.000 ký tự tiếng Anh thì còn chỗ cho khoảng 80-100 dòng code. Bài dài hơn phải chia phần. |
| Hỏi tiếp trong cùng cuộc trò chuyện thì lịch sử cũng bị tính token. | Mỗi phần, mỗi lần sửa đều **mở cuộc trò chuyện mới**. Đừng bảo "continue" khi đã bị cắt, vì chỉ còn khoảng 130 token. |
| Token đầu tiên tới sau 0,4 giây; tốc độ khoảng 22 token/giây; phần suy luận không hiện ra và không tính token. | Chatbot gần như không tự lập kế hoạch: phải viết sẵn các bước, tên hàm, câu lệnh. Một câu trả lời 1.300 token mất khoảng 60 giây. |
| Cùng một prompt, mỗi lần trả lời một khác: lần này vừa đủ token, lần sau có thể bị cắt. | Prompt sát ngưỡng là rủi ro. Bị cắt thì gửi lại trong cuộc mới, hoặc rút gọn prompt. |
| Hay bỏ qua yêu cầu "at most N lines", "no comments". | Đừng dựa vào giới hạn số dòng. Dòng `ANSWER` chỉ định dòng code đầu tiên có tác dụng hơn (xem mục 3). |
| Tiếng Anh tốn ít token hơn tiếng Việt. | Viết prompt bằng tiếng Anh. Chỉ giữ tiếng Việt cho tên nhãn và dữ liệu. |

## 2. Quy trình trong phòng thi

1. **Phút đầu:** gõ `dau_file.py` (35 dòng, nên thuộc lòng) thành phần đầu của `main.py`. Có file này thì chatbot không phải viết import, seed, chọn device, autocast. Đây là nhóm lỗi hay gặp nhất khi để chatbot tự viết.
2. **Đọc đề, viết prompt** theo `khung_1_phan.txt`. Nếu ước code dài hơn khoảng 90 dòng (mạng nơ-ron có vòng train riêng, tiền xử lý dài), dùng `khung_2_phan.txt`. Hàm metric tự viết (BLEU, cF1, tIoU...) nên hỏi riêng một cuộc, kèm `assert`.
3. **Nhận code:** dán ngay sau phần đầu file. Đọc lướt 30 giây để bắt các đoạn chatbot tự thêm. Chạy thử nhanh với 1 epoch hoặc một phần nhỏ dữ liệu.
4. **Lỗi vặt** (`NameError`, thiếu import, thiếu hằng số): **sửa tay** trong 10 giây, đừng tốn token hỏi lại.
5. **Lỗi logic** (kết quả sai, `KeyError` trong thuật toán): dùng `khung_sua_loi.txt`. Gửi đúng đoạn lỗi, dòng lỗi, ý nghĩa biến và mục tiêu. Một lần sửa tốn 300-600 token. **Đọc lại bản sửa trước khi dán:** trong 2 lần thử, chatbot sửa đúng 1 lần và sửa một nửa 1 lần.
6. **Kết quả kém bất thường** (macro-F1 gần mức đoán ngẫu nhiên): kiểm tra cache cũ (`.npy`, `best.pt`) trước khi nghi model. Kiểm tra cả định dạng file nộp: tên lớp hay số, id gốc hay số thứ tự dòng.

## 3. Mười một bài học từ các lần thử (nên thuộc)

1. **Ghi câu lệnh cụ thể cho phần đọc dữ liệu.** Prompt chỉ nói "load every image" thì chatbot dùng `np.load` cho file PNG. Prompt ghi `np.array(Image.open(path).convert("RGB"))` thì đúng.
2. **Ghi chữ ký hàm và dạng giá trị trả về.** Bản đầu bài d05 lệch tuple: `get_neighbors` trả 4 giá trị, chỗ gọi nhận 3. Bản đầu bài d30 gọi `ColumnTransformer` thiếu danh sách cột. Ghi sẵn `dijkstra(g, src)` trả về `(dist, par)` với `par[(r, c)] = ((pr, pc), move)` thì hết lỗi.
3. **Tách phần đầu file.** Câu HEADER phải viết đúng dạng "already at the top of the file, do not repeat it: import ..." và liệt kê đủ tên. Thử viết gọn kiểu "it imports torch, nn, ..." thì chatbot lại chép cả phần đầu, tốn thêm khoảng 25 dòng.
4. **Chỉ định dòng đầu tiên** (`ANSWER: ... Its first line must be: train = pd.read_csv(...)`). Cách này cắt hẳn phần import chép lại và lời dẫn, giúp 3 prompt từ bị cắt chuyển thành vừa đủ token.
5. **Viết sẵn các đoạn dễ sai:**
   - truy vết quy hoạch động;
   - ghi file nộp: id gốc, tên lớp chứ không phải số;
   - ghép kết quả về đúng vị trí sau khi sắp xếp theo độ dài.

   Chatbot đã sai cả ba chỗ này khi chỉ được mô tả bằng lời: d04 ghi `id` là số thứ tự dòng, d28 ghi nhãn là số.
6. **Sắp đúng thứ tự các bước.** d28 A2 từng mã hoá câu trước khi xây xong vocab. Khi chia 2 phần và ghi từng câu lệnh, code chỉ còn 38 + 35 dòng và chạy đúng ngay.
7. **Bài dài thì chia 2 phần, nối bằng CONTRACT:** danh sách tên biến, hàm, lớp kèm kiểu, chép nguyên văn vào cả hai prompt. Phần 1 bị cắt thì dời một hàm sang phần 2 (mục `0.` trong khung).
8. **Liệt kê sẵn import trong `dau_file.py`.** Tên nào không có trong HEADER thì chatbot hay dùng mà quên import (`pad_sequence`, `StratifiedGroupKFold`).
9. **Đọc lướt phần chatbot tự thêm.** Ví dụ d23 tự thêm `isinstance(model, make_pipeline)`, câu lệnh này luôn lỗi.
10. **Xoá cache khi code đổi.** d23 từng được macro-F1 0,16 chỉ vì đọc lại `feat_train.npy` do phiên bản code trước tạo ra. Xoá cache xong thì đạt 0,865.
11. **Metric tự viết phải có `assert` tự kiểm tra.** BLEU do chatbot viết trong d29 cho ra 3,6, trong khi BLEU không thể vượt 1. Mô hình vẫn qua ngưỡng (test 0,910), nhưng checkpoint được chọn theo một metric sai mà không có lỗi nào báo; khi BLEU đúng, test lên 0,977. Cách sửa: viết metric trong một cuộc trò chuyện riêng, chỉ một hàm, kèm 3 dòng `assert`: trùng khớp được 1, không khớp được 0, khớp một phần nằm giữa 0 và 1 (xem `mau/d29_seq2seq_bleu.txt`). Bản viết lại khớp nltk tới 1e-16.

## 4. Kết quả thử với các bài luyện (28/11, chatbot mô phỏng)

Cách làm: gửi prompt trong `mau/` (mỗi phần một cuộc trò chuyện mới), ghép với `dau_file.py`, chạy bộ chấm `check_NN.py` trên CPU. Cột "Sửa tay" là số dòng phải tự sửa ở lần cuối.

| Bài | Prompt cuối | Token (mỗi cuộc) | Các lần trước hỏng vì | Sửa tay | Điểm | Ngưỡng |
|---|---|---|---|---|---|---|
| d28 A - TF-IDF + logistic | `d28a_tfidf.txt` | 760 | - | 0 | macro-F1 0,908 | 0,87 |
| d28 B - BM25 tự viết | `d28b_bm25.txt` | 878 | - | 0 | Recall@1 0,930 | 0,85 |
| d30 - dữ liệu bảng, stacking | `d30_du_lieu_bang.txt` | 1.491 | `ColumnTransformer` thiếu cột | 0 | AUROC 0,749 | 0,73 |
| d21 - fine-tune ResNet-18 | `d21_phan_loai_anh.txt` | 1.759 | `np.load` file PNG; thiếu import `Image`; thiếu `dev` | 0 | macro-F1 0,955 | 0,95 |
| d05 A - Dijkstra + Held-Karp | `d05a_duong_di.txt` | 1.677 | lệch tuple; thiếu `heapq`; truy vết sai | 1 dòng (`MOVES`) | 1,000 | 0,99 |
| d04 A - DimABSA, hồi quy VA | `d04a_dimabsa_va.txt` | 1.602 | `id` thành số dòng; không gọi `seed_all`; 2 lần bị cắt | 0 | RMSE_VA 0,528 | ≤ 0,70 |
| d23 - ResNet-50 đóng băng + CV nhóm | `d23_dac_trung_dong_bang.txt` | 1.422 | quên tạo backbone; thiếu import | 1 dòng + xoá cache | macro-F1 0,865 | 0,85 |
| d28 A2 - TextCNN tự train | `d28a2_textcnn_p1/p2.txt` | 1.465 + 1.123 | bản 1 phần: bị cắt, hoặc sai thứ tự vocab và batching | 0 | macro-F1 0,897 | 0,85 |
| d01 - âm thanh, log-mel, CNN | `d01_am_thanh_p1/p2.txt` | 1.712 + 1.634 | bản 1 phần và bản 2 phần đầu: bị cắt; phần 2 thiếu `forward` vì prompt không nhắc | 4 dòng (`wavfile`, `with torch.manual_seed`, `AdamW`, `OneCycleLR` thiếu tiền tố) | macro-F1 0,895 | 0,85 |
| d29 - Transformer dịch máy | `d29_seq2seq_p1.txt`, `d29_seq2seq_bleu.txt`, `d29_seq2seq_p2.txt` (3 cuộc) | 1.599 + 781 + 1.693 | bản 1 phần và phần 1 nhiều lần: bị cắt; BLEU tự viết sai (ra 3,6, test chỉ 0,910); mặt nạ truyền nhầm tham số | 1 dòng (sau khi dùng prompt sửa lỗi) | BLEU-4 0,977 | 0,90 |
| Sửa đoạn truy vết Held-Karp | `sua_loi_held_karp.txt` | 317 | - | 0 | đúng 200/200 ca thử | - |
| Sửa mặt nạ Transformer (d29) | `sua_loi_mat_na_transformer.txt` | 620 | - | 1 dòng: chatbot truyền đúng tên tham số nhưng vẫn tính mặt nạ nguồn sau embedding | chạy được, xem dòng d29 | - |

Nhận xét:
- Bài sklearn ngắn (TF-IDF, BM25, bảng) đạt ngay khi prompt ghi đủ câu lệnh.
- Bài có mạng nơ-ron cần phần đầu file chuẩn, dòng `ANSWER` và thường phải chia 2 phần.
- Lỗi còn sót lại hầu hết là lỗi vặt sửa tay trong vài giây.
- Prompt cuối mỗi bài đều được viết lại 2-4 lần. Lần viết đầu tiên của bạn trong phòng thi cũng sẽ như vậy, nên hãy luyện để viết đúng ngay từ đầu.

## 5. Cách luyện

**Chuẩn bị:** bật chatbot mô phỏng, rồi đặt địa chỉ của nó vào biến môi trường:

```
$env:CHATBOT_URL = "https://<địa-chỉ>/chatbot"
```

**Mỗi buổi 45 phút, bấm giờ:**
1. Chọn một bài trong `luyen_tap/CV` **chưa có** prompt mẫu: d22, d24, d25, d26, d27, d02, d03, d04 B, d05 B. Chạy `python tao_du_lieu.py` trong thư mục bài.
2. Viết prompt theo khung, **không mở** `mau/`. Ước số token trước khi gửi: 1 token khoảng 4 ký tự tiếng Anh, tổng phải dưới 2.000.
3. Gửi và ghép code:

   ```
   python on_tap/prompt_llm/hoi_llm.py luyen_tap/CV/d26_video/bai_lam.py prompt_p1.txt prompt_p2.txt --dau
   ```

4. Chấm: `python check_NN.py bai_lam.py`. Sửa tay lỗi vặt; lỗi logic thì dùng khung sửa lỗi.
5. Ghi lại:
   - số lần gửi;
   - token mỗi lần;
   - số dòng sửa tay;
   - một câu "lần sau viết prompt khác đi thế nào".
6. So với prompt mẫu cùng dạng trong `mau/` để học cách ghi câu lệnh.

**Mục tiêu trước 7/12:** mỗi bài đạt ngưỡng với **tối đa 2 lần gửi prompt** và **tối đa 3 dòng sửa tay**. Bài sklearn trong 10 phút, bài mạng nơ-ron trong 25 phút.

Nhắc lại luật: mọi code do chatbot viết vẫn phải tuân luật pretrained của vòng thi. Ở vòng khu vực: không có pretrained NLP; CV chỉ dùng backbone ImageNet trong danh sách BTC. Chung kết dự kiến tương tự, nhưng phải kiểm tra lại đề. Ghi luật này vào dòng `TASK`.
