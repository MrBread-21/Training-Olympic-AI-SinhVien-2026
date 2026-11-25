"""Sinh <Bài>/KAGGLE.md (nội dung tách sẵn từng ô khi tạo cuộc thi Kaggle) từ README.md của bài và vài thông tin riêng.
Description = phần Overview của README bỏ khối "Luật"; Rules = khối "Luật"; Evaluation, Submission File, Dataset Description
lấy nguyên từ các mục cùng tên. Điểm baseline đọc từ host/min_score.txt.
    python tools/sinh_kaggle_md.py ViQuAD_ExtractiveQA HomeCredit_Tabular ..."""
import re
import sys

from chung import GOC

TT = {  # bài: (Title, URL, Subtitle, Overview ngắn, metric, chiều, Data license, Citation)
    "ViQuAD_ExtractiveQA": (
        "OlpAI thử - Hỏi đáp trích xuất tiếng Việt", "pka-olpai-test-viquad-qa",
        "Chép câu trả lời từ đoạn văn Wikipedia tiếng Việt, hoặc để trống nếu không có (dạng bài VIQA).",
        "Cho đoạn văn và câu hỏi, trích cụm từ trả lời hoặc để trống khi đoạn văn không có câu trả lời (khoảng 30% câu hỏi). Chấm bằng F1 theo từ. Không dùng model ngôn ngữ pretrained.",
        "qa_f1", "cao hơn là tốt hơn", "Other (specified in description): UIT-ViQuAD 2.0, chỉ dùng cho học tập và nghiên cứu.",
        "Nguyen và cộng sự, VLSP 2021 - ViQuAD 2.0: Vietnamese Machine Reading Comprehension. Bản Hugging Face taidng/UIT-ViQuAD2.0."),
    "HomeCredit_Tabular": (
        "OlpAI thử - Dự đoán vỡ nợ (dữ liệu bảng)", "pka-olpai-test-homecredit",
        "Dự đoán khách hàng vỡ nợ từ hồ sơ vay và lịch sử tín dụng ở nhiều bảng; test là các tháng sau train.",
        "Dự đoán khả năng vỡ nợ từ hồ sơ vay, lịch sử tín dụng ở tổ chức khác và các khoản vay trước (3 bảng nối bằng id). Train là 01/2019-06/2020, test là 07-12/2020. Chấm bằng AUROC.",
        "auroc", "cao hơn là tốt hơn", "Other (specified in description): dữ liệu cuộc thi Home Credit Default Risk, chỉ dùng học tập nội bộ, không phát tán lại.",
        "Home Credit Group - Home Credit Default Risk (Kaggle, 2018). Bản Hugging Face mohameddhameem/home-credit-default-risk."),
    "Robot_PathPlanning": (
        "OlpAI thử - Robot đi qua mọi điểm mốc", "pka-olpai-test-robot-path",
        "Tìm đường rẻ nhất cho robot trên lưới có địa hình và tường: ghé mọi điểm mốc rồi về đích.",
        "Robot xuất phát ở S, ghé mọi điểm mốc theo thứ tự tuỳ chọn rồi dừng ở G, tổng chi phí địa hình nhỏ nhất. Điểm mỗi bản đồ = chi phí tối ưu / chi phí của bạn.",
        "path_score", "cao hơn là tốt hơn", "CC0: bản đồ do người dựng đề sinh ngẫu nhiên.",
        "Bản đồ sinh ngẫu nhiên cho buổi luyện tập OlpAI'26."),
    "KTVIC_Captioning": (
        "OlpAI thử - Sinh caption tiếng Việt cho ảnh", "pka-olpai-test-ktvic-caption",
        "Viết một câu tiếng Việt mô tả mỗi ảnh đời sống hằng ngày (dạng bài chung kết OlpAI 2025).",
        "Sinh một câu mô tả tiếng Việt cho mỗi ảnh, chấm bằng CIDEr-D với 5 câu tham chiếu. Phần ảnh chỉ dùng backbone ImageNet của BTC, phần sinh câu train từ đầu.",
        "cider_d", "cao hơn là tốt hơn", "Other (specified in description): KTVIC, chỉ dùng cho học tập và nghiên cứu.",
        "Pham và cộng sự, 2024 - KTVIC: A Vietnamese Image Captioning Dataset on the Life Domain. Bản Hugging Face ai-enthusiasm-community/KTVIC."),
    "SpeechCommands_KWS": (
        "OlpAI thử - Nhận dạng từ khoá trong âm thanh", "pka-olpai-test-speech-kws",
        "Nhận dạng 10 lệnh nói và từ lạ trong đoạn âm thanh 1 giây, test là người nói mới và có tạp âm.",
        "Phân loại đoạn ghi âm 1 giây thành 10 lệnh (yes, no, up, down, left, right, on, off, stop, go) hoặc unknown. Test là người nói chưa gặp, 70% có tạp âm. Chấm bằng macro-F1.",
        "macro_f1", "cao hơn là tốt hơn", "CC BY 4.0",
        "Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209. Bản Hugging Face mteb/speech-commands-mini."),
    "CIFAR_Adversarial": (
        "OlpAI thử - Tấn công đối kháng CIFAR-10", "pka-olpai-test-cifar-adversarial",
        "Sửa ảnh không quá 8 mức xám để lừa mô hình đã công bố và một mô hình bị giấu (dạng bài chung kết OlpAI 2025).",
        "Tạo ảnh đối kháng lệch ảnh gốc không quá 8/255 để mô hình A (công bố) và mô hình B (giấu kín, cùng kiến trúc) đoán sai. Mỗi ảnh hợp lệ được 0,5 cho mỗi mô hình bị lừa.",
        "adv_success", "cao hơn là tốt hơn", "Other (specified in description): CIFAR-10, dùng cho học tập và nghiên cứu.",
        "Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Bản Hugging Face uoft-cs/cifar10."),
}

TIMELINE = """- Bắt đầu: do người tổ chức thông báo.
- Giờ 0-5: làm bài, tối đa 20 lần nộp, chỉ có public test.
- Giờ thứ 6: công bố mật khẩu `private_test.zip`; dự đoán private, viết `main.py`, chọn bài nộp cuối.
- Kết thúc: 6 tiếng sau giờ bắt đầu; bảng private được mở ngay sau đó."""


def muc(md, ten):
    m = re.search(rf"^## {re.escape(ten)}\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


for b in sys.argv[1:]:
    title, url, sub, ov, metric, chieu, lic, cit = TT[b]
    md = (GOC / b / "README.md").read_text(encoding="utf-8")
    tong = muc(md, "Overview")
    luat = re.search(r"^\*\*Luật.*?(?=\n\n[^-\s]|\Z)", tong, re.S | re.M)
    rules = re.sub(r"^\*\*Luật[^\n]*\*\*:?\s*", "", luat.group(0)).strip() if luat else ""
    gioi_thieu = md.split("\n## ", 1)[0].split("\n", 1)[1].strip()  # đoạn giữa tiêu đề và mục đầu tiên
    desc = gioi_thieu + "\n\n" + (tong.replace(luat.group(0), "") if luat else tong).strip()
    desc = re.sub(r"\n{3,}", "\n\n", desc)
    if not rules.startswith("-"):
        rules = "- " + rules
    rules += "\n- Không mở `private_test.zip` trước khi mật khẩu được công bố.\n- Tối đa 20 lần nộp; chọn bài nộp cuối để chấm private."
    try:
        diem = dict(l.split()[:2] for l in (GOC / b / "host/min_score.txt").read_text(encoding="utf-8").splitlines() if l.strip())
        moc = f"public {diem['Public']}, private {diem['Private']}"
    except (FileNotFoundError, KeyError, ValueError):
        moc = "xem host/min_score.txt"
    out = f"""# {b} - nội dung từng ô trên Kaggle

## Title

{title}

## Competition URL

{url}

## Subtitle

{sub}

## Overview

{ov}

## Description

{desc}

## Evaluation

{muc(md, "Evaluation")}

## Submission File

{muc(md, "Submission File")}

## Timeline

{TIMELINE}

## Rules

{rules}

## Citation

{cit}

## Dataset Description

{muc(md, "Data")}

## Data license

{lic}

## Files để tải lên

- Data: `{b}_data.zip`
- Metric notebook: dán toàn bộ `host/metric.py` (hàm `{metric}`), {chieu}
- Solution: `host/solution.csv` (cột id `id`, cột chia `Usage`)
- Sample submission: `upload/sample_submission.csv`
- Mật khẩu private test: `host/mat_khau_private_test.txt`, công bố ở tiếng cuối
- Kiểm tra bằng sandbox submission: nộp `host/nop_dung_het.csv` (nếu có) phải ra điểm tối đa; nộp `submission.csv` của baseline phải gần `host/min_score.txt` ({moc})
"""
    (GOC / b / "KAGGLE.md").write_text(out, encoding="utf-8")
    print(b, "KAGGLE.md", len(out), "ký tự; rules", rules.count("\n") + 1, "dòng")
