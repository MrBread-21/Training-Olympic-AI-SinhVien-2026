"""Tạo dữ liệu bài 28 (tiếng Việt giả lập). Bẫy: train gần như toàn NFC nhưng 60% văn bản test và câu hỏi dạng NFD,
20% mất dấu, lỗi gõ, từ của chủ đề khác.
A: phân loại chủ đề (3000 train, 1000 test). B: truy xuất đoạn văn (400 đoạn, 300 câu hỏi). CPU khoảng 2 giây."""
import random
import unicodedata
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = random.Random(2801)
D, A = HERE / "data", HERE / ".dap_an"
D.mkdir(exist_ok=True); A.mkdir(exist_ok=True)
TU = {
    "the_thao": "bóng đá|cầu thủ|huấn luyện viên|trận đấu|bàn thắng|đội tuyển|giải vô địch|sân vận động|thủ môn|tiền đạo|ghi bàn|hiệp hai|trọng tài|thẻ đỏ|cổ động viên|vòng loại|huy chương|vận động viên|điền kinh|bơi lội|quần vợt|kỷ lục|chung kết|tỉ số|phạt đền",
    "kinh_te": "lãi suất|ngân hàng|chứng khoán|cổ phiếu|lạm phát|tăng trưởng|xuất khẩu|nhập khẩu|doanh nghiệp|lợi nhuận|doanh thu|thị trường|đầu tư|tỷ giá|ngân sách|thuế|giá vàng|bất động sản|vốn|cổ đông|kinh doanh|nợ xấu|tín dụng|quý ba|GDP",
    "cong_nghe": "điện thoại|trí tuệ nhân tạo|phần mềm|ứng dụng|máy tính|internet|dữ liệu|bảo mật|lập trình|chip|công nghệ|mạng xã hội|robot|thuật toán|máy chủ|điện toán đám mây|tin tặc|phiên bản|cập nhật|thiết bị|màn hình|pin|camera|khởi nghiệp|5G",
    "suc_khoe": "bệnh viện|bác sĩ|bệnh nhân|vắc xin|điều trị|sức khỏe|dinh dưỡng|tập thể dục|thuốc|triệu chứng|phẫu thuật|dịch bệnh|virus|huyết áp|tiểu đường|giấc ngủ|khám bệnh|y tế|cấp cứu|tim mạch|ung thư|miễn dịch|sốt|ho|chế độ ăn",
    "giao_duc": "học sinh|giáo viên|trường học|kỳ thi|đại học|tuyển sinh|sinh viên|học bổng|chương trình học|sách giáo khoa|điểm chuẩn|lớp học|tốt nghiệp|bài giảng|học phí|nghiên cứu|thạc sĩ|tiến sĩ|mầm non|đề thi|môn toán|ngữ văn|ôn tập|giảng đường|thi tốt nghiệp",
}
TU = {k: v.split("|") for k, v in TU.items()}
CHUNG = ("hôm nay|theo|cho biết|trong|năm nay|nhiều|người|đã|sẽ|được|các|những|với|tại|sau khi|vào|rất|khá|mới|đang|cũng|của|"
         "này|một|hai|ba|lần|đầu tiên|tháng|tuần qua|thành phố|Hà Nội|TP.HCM|Việt Nam|chuyên gia|cho rằng|có thể|tiếp tục").split("|")


def bo_dau(s):
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))).replace("đ", "d").replace("Đ", "D")


def bay(s, p_nfd):
    """Thêm nhiễu: lỗi gõ, mất dấu, NFD với xác suất p_nfd."""
    if R.random() < 0.15:
        i = R.randrange(len(s)); s = s[:i] + s[i + 1:]
    if R.random() < 0.2: s = bo_dau(s)
    if R.random() < p_nfd: s = unicodedata.normalize("NFD", s)
    return s


rows = []
for i in range(4000):
    k = list(TU)[i % 5]
    w = R.sample(TU[k], 2) + R.sample(CHUNG, R.randint(5, 9))
    if R.random() < 0.35: w += R.sample(TU[R.choice([c for c in TU if c != k])], R.randint(1, 2))   # từ gây nhiễu
    R.shuffle(w); s = " ".join(w); s = s[0].upper() + s[1:] + "."
    rows.append((f"v{i:04d}", s, k))
df = pd.DataFrame(rows, columns=["id", "text", "label"]).sample(frac=1, random_state=0)
df["text"] = [bay(t, 0.05 if j < 3000 else 0.6) for j, t in enumerate(df.text)]
df[:3000].to_csv(D / "a_train.csv", index=False)
df[3000:][["id", "label"]].sort_values("id").to_csv(A / "a_test.csv", index=False)
df[3000:][["id", "text"]].to_csv(D / "a_test.csv", index=False)

AM = "Minh|Phát|Hoàng|Long|Thịnh|An|Việt|Tân|Hưng|Phúc|Đại|Nam|Bình|Quang|Thành|Gia|Kim|Sao|Hồng|Thái".split("|")
TP = ("Hà Nội|Hải Phòng|Đà Nẵng|Huế|Cần Thơ|Nha Trang|Vinh|Quy Nhơn|Đà Lạt|Vũng Tàu|Biên Hòa|Thái Nguyên|Nam Định|Thanh Hóa|"
      "Hạ Long|Buôn Ma Thuột|Pleiku|Phan Thiết|Rạch Giá|Cà Mau|Long Xuyên|Mỹ Tho|Bắc Ninh|Hải Dương|Lào Cai").split("|")
SP = ("xi măng|thép|gạo|cà phê|điện thoại|phần mềm|dệt may|nước mắm|giày dép|đồ gỗ|sữa|bánh kẹo|phân bón|thuốc|ô tô điện|"
      "pin mặt trời|gốm sứ|kính|giấy|nhựa|xe đạp|tủ lạnh|chè|hạt điều").split("|")
HO = "Nguyễn|Trần|Lê|Phạm|Hoàng|Vũ|Đặng|Bùi|Đỗ|Ngô".split("|")
seen, P, Q = set(), [], []
while len(P) < 400:
    f = (" ".join(R.sample(AM, 2)), R.choice(TP), R.choice(SP), R.randint(1990, 2023), f"{R.choice(HO)} Văn {R.choice(AM)}")
    if f[1:3] in seen: continue                                   # mỗi cặp (thành phố, sản phẩm) chỉ một công ty
    seen.add(f[1:3]); pid = f"p{len(P):03d}"
    P.append((pid, f"Công ty {f[0]} được thành lập năm {f[3]} tại {f[1]}, chuyên sản xuất {f[2]}. "
                   f"Giám đốc hiện nay là ông {f[4]}. Năm nay doanh thu của công ty tăng trưởng khá so với năm trước."))
    if len(Q) < 300:
        mau = R.choice([f"Công ty nào sản xuất {f[2]} ở {f[1]}?", f"Doanh nghiệp ở {f[1]} chuyên làm {f[2]} do ai làm giám đốc?",
                        f"Hãng {f[2]} tại {f[1]} thành lập năm nào?", f"Ông giám đốc của nơi sản xuất {f[2]} tại {f[1]} là ai?"])
        Q.append((f"q{len(Q):03d}", bay(mau, 0.6), pid))
pd.DataFrame(P, columns=["id", "text"]).to_csv(D / "b_corpus.csv", index=False)
q = pd.DataFrame(Q, columns=["id", "question", "passage_id"]).sample(frac=1, random_state=1)
q[["id", "question"]].to_csv(D / "b_questions.csv", index=False)
q[["id", "passage_id"]].sort_values("id").to_csv(A / "b_test.csv", index=False)
print("xong: A 3000 train / 1000 test; B 400 đoạn / 300 câu hỏi")
