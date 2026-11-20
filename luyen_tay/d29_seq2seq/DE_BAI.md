# Bài 29 (29/11) - Seq2seq Transformer từ đầu (dịch Anh sang Việt)

**Thời gian mục tiêu: ≤ 50 phút.**

## Đọc trước khi làm

Đọc trong khoảng 30-45 phút trước khi bấm giờ; ghi ra giấy 3-5 ý chính (công thức, tham số, bẫy). Lúc làm bài thì đóng hết.

- **Đọc chính**: [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (Harvard NLP): mục model architecture, masking (subsequent mask), batches and masking, greedy decoding, label smoothing.
- **PyTorch**: [nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) (ý nghĩa `src_key_padding_mask`, `tgt_mask`, `memory_key_padding_mask`, `batch_first`), [F.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) (`ignore_index`, chiều lớp phải ở trục 1).
- **Bài báo**:
  - Vaswani và cộng sự, *Attention Is All You Need* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), mục 3 (kiến trúc) và 5.4 (label smoothing).
  - Beam search có length penalty: Wu và cộng sự, GNMT [arXiv:1609.08144](https://arxiv.org/abs/1609.08144), mục 7.
  - BLEU: Papineni và cộng sự, ACL 2002 ([P02-1040](https://aclanthology.org/P02-1040/)), mục 2 (n-gram cắt, brevity penalty). chrF: Popović, WMT 2015 ([W15-3049](https://aclanthology.org/W15-3049/)).
- **Trong máy**: `../../ON_TAP.md` phần 1.4 (đề 2025 vòng khu vực là dịch máy).

## Bài 1

- **Đầu vào**: `data/train.csv` (`en,vi`, 12000 cặp), `data/val.csv` (500 cặp), `data/test.csv` (`id,en`, 1000 câu). Ngữ pháp có: tính từ đứng sau danh từ, lượng từ (con, quyển, chiếc...), **số đếm 1-99** (21 thành "hai mươi mốt", 15 thành "mười lăm"), "my X" thành "X của tôi", thì quá khứ "đã", tân ngữ kép nối bằng "and".
- **Đầu ra**: `submission.csv` (`id,vi`).
- **Metric**: BLEU-4 corpus (tách từ theo khoảng trắng, chữ thường), **ngưỡng 0.90**; in thêm chrF.
- **Khung bắt buộc**:
  1. Vocab nguồn / đích (`<pad>=0, <unk>, <bos>, <eos>`), padding bằng `pad_sequence`, bucketing theo độ dài câu nguồn.
  2. `nn.Transformer(batch_first=True)`, embedding + positional embedding; **mặt nạ nhân quả** (`generate_square_subsequent_mask`) và **mặt nạ padding** cho cả nguồn, đích, memory.
  3. Teacher forcing: đầu vào decoder = `t[:, :-1]`, nhãn = `t[:, 1:]`; `cross_entropy(..., ignore_index=0, label_smoothing=0.1)`; `logits.transpose(1, 2)`; clip gradient.
  4. Mỗi epoch: BLEU trên val (greedy), ngoài vòng batch; lưu checkpoint tốt nhất.
  5. Greedy theo batch (dừng khi mọi câu có `<eos>`); **beam search** (k = 4, length penalty) và so với greedy trên 100 câu val.
  6. Tự viết BLEU-4 (n-gram cắt theo tham chiếu, brevity penalty).
- Viết đúng thì val BLEU khoảng 0.34 sau epoch đầu và khoảng 0.99 sau 10 epoch (khoảng 1.5 phút trên GTX 1650). Sai mặt nạ hoặc lệch dịch chuỗi đích sẽ cho BLEU rất thấp.

## Bài 2 - Viết lại nhanh bài 28 (≤ 35 phút)

Trong `d28_nlp`: viết lại cả ba file, chấm `python check_28.py viet_lai_a.py viet_lai_a2.py viet_lai_b.py`.

## Cách chạy

```
python tao_du_lieu.py
python check_29.py bai_lam.py
```
