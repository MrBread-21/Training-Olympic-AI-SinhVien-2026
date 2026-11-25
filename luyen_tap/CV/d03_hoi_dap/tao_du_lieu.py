"""Tạo dữ liệu bài 03 (tiếng Việt giả lập, CPU vài giây): đoạn tiểu sử nhân vật hư cấu + câu hỏi, khoảng 25% câu hỏi
không có đáp án trong đoạn (hỏi về thông tin đoạn văn không nhắc tới). Bẫy: nhiều năm và số gây nhiễu trong cùng đoạn,
câu hỏi diễn đạt nhiều cách, 40% đoạn văn và câu hỏi của test ở dạng Unicode NFD.
Train: 700 đoạn, test: 260 đoạn khác (chia theo đoạn)."""
import random
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
R = random.Random(303)
D, A = HERE / "data", HERE / ".dap_an"
D.mkdir(exist_ok=True); A.mkdir(exist_ok=True)

HO = "Nguyễn|Trần|Lê|Phạm|Hoàng|Vũ|Đặng|Bùi|Đỗ|Ngô|Dương|Lý|Trịnh|Đinh|Mai".split("|")
DEM = "Văn|Thị|Minh|Đức|Thanh|Ngọc|Quang|Hữu|Thu|Bảo|Gia|Kim".split("|")
TEN = "An|Bình|Cường|Dũng|Giang|Hà|Hải|Hiếu|Hoa|Hùng|Khánh|Lan|Linh|Long|Mai|Nam|Ngân|Phong|Phúc|Quân|Sơn|Tâm|Thảo|Trang|Tuấn|Vy|Yến".split("|")
TINH = ("Hà Nội|Hải Phòng|Nam Định|Thái Bình|Nghệ An|Hà Tĩnh|Quảng Bình|Huế|Đà Nẵng|Quảng Nam|Bình Định|Khánh Hòa|"
        "Đắk Lắk|Lâm Đồng|Cần Thơ|An Giang|Bến Tre|Cà Mau|Lào Cai|Thanh Hóa").split("|")
NGHE = {"nhà vật lý": "các công trình về quang học", "nhà văn": "tiểu thuyết", "kiến trúc sư": "các toà nhà",
        "nhạc sĩ": "các ca khúc", "kỹ sư cầu đường": "các cây cầu", "nhà sinh học": "các nghiên cứu về lúa gạo",
        "hoạ sĩ": "các bức tranh sơn mài", "nhà toán học": "các định lý về số nguyên tố"}
TAC_PHAM = ("Mùa gặt trên sông|Ánh sáng phương Nam|Những ngọn đèn xa|Đường về làng cũ|Bản giao hưởng biển|Cánh đồng gió|"
            "Thành phố trong sương|Tiếng vọng núi rừng|Dòng chảy thời gian|Hạt gạo vàng|Cầu vồng trên phố|Ngôi nhà gỗ").split("|")
TRUONG = "Đại học Bách khoa|Đại học Sư phạm|Đại học Mỹ thuật|Nhạc viện|Đại học Khoa học Tự nhiên|Đại học Xây dựng".split("|")
GIAI = "giải thưởng Hồ Chí Minh|giải thưởng Nhà nước|huân chương Lao động|giải Bông Sen Vàng|giải thưởng Tạ Quang Bửu".split("|")


def nguoi():
    ten = f"{R.choice(HO)} {R.choice(DEM)} {R.choice(TEN)}"
    sinh = R.randint(1920, 1985)
    return dict(ten=ten, goi=ten.split()[-1], sinh=str(sinh), que=R.choice(TINH), nghe=R.choice(list(NGHE)),
                truong=R.choice(TRUONG), tn=str(sinh + R.randint(21, 26)), tp=R.choice(TAC_PHAM),
                nam_tp=str(sinh + R.randint(28, 45)), giai=R.choice(GIAI), nam_giai=str(sinh + R.randint(40, 60)),
                so=str(R.randint(3, 60)), hoc_tro=str(R.randint(10, 300)))


def doan(f):
    """Ghép các câu (một số câu bị bỏ ngẫu nhiên). Trả (văn bản, {loại thông tin: (đáp án, vị trí ký tự)})."""
    cau = [
        ("sinh", [f"{f['ten']} sinh năm {f['sinh']} tại {f['que']}.", f"{f['ten']} ({f['sinh']}) là người con của đất {f['que']}."]),
        ("nghe", [f"Ông là {f['nghe']} nổi tiếng với {NGHE[f['nghe']]}.", f"Sự nghiệp của ông gắn với nghề {f['nghe']}."]),
        ("truong", [f"Năm {f['tn']}, ông tốt nghiệp {f['truong']}.", f"Ông học tại {f['truong']} và ra trường năm {f['tn']}."]),
        ("tp", [f"Tác phẩm tiêu biểu nhất của ông là \"{f['tp']}\" ra mắt năm {f['nam_tp']}.",
                f"Năm {f['nam_tp']}, ông công bố \"{f['tp']}\", được xem là tác phẩm tiêu biểu nhất."]),
        ("giai", [f"Năm {f['nam_giai']}, {f['goi']} được trao {f['giai']}.", f"Ông nhận {f['giai']} vào năm {f['nam_giai']}."]),
        ("so", [f"Trong đời mình, ông để lại {f['so']} công trình và đào tạo {f['hoc_tro']} học trò.",
                f"Ông có {f['so']} công trình đã công bố, cùng {f['hoc_tro']} học trò."]),
    ]
    giu = [c for c in cau if c[0] in ("sinh", "nghe") or R.random() < 0.7]
    van, vt = "", {}
    for loai, mau in giu:
        s = R.choice(mau)
        vt[loai] = (len(van), s)               # vị trí bắt đầu và nội dung câu chứa thông tin này
        van += s + " "
    return van.strip(), vt


HOI = {  # loại câu hỏi: (thông tin cần có, đáp án, các mẫu câu hỏi)
    "nam_sinh": ("sinh", "sinh", ["{ten} sinh năm nào?", "{ten} ra đời vào năm nào?", "Năm sinh của {ten} là năm nào?"]),
    "que": ("sinh", "que", ["{ten} quê ở đâu?", "{ten} sinh ra ở tỉnh nào?", "Quê hương của {ten} là ở đâu?"]),
    "nghe": ("nghe", "nghe", ["{ten} làm nghề gì?", "Nghề nghiệp của {ten} là gì?"]),
    "truong": ("truong", "truong", ["{ten} tốt nghiệp trường nào?", "{goi} học ở trường nào?"]),
    "nam_tn": ("truong", "tn", ["{ten} tốt nghiệp năm nào?", "{goi} ra trường vào năm nào?"]),
    "tp": ("tp", "tp", ["Tác phẩm tiêu biểu nhất của {ten} là gì?", "{goi} nổi tiếng nhất với tác phẩm nào?"]),
    "nam_tp": ("tp", "nam_tp", ["Tác phẩm tiêu biểu của {ten} ra mắt năm nào?", "{goi} công bố tác phẩm tiêu biểu vào năm nào?"]),
    "giai": ("giai", "giai", ["{ten} được trao giải thưởng gì?", "{goi} đã nhận giải thưởng nào?"]),
    "nam_giai": ("giai", "nam_giai", ["{ten} nhận giải thưởng vào năm nào?", "Năm nào {goi} được trao giải?"]),
    "so": ("so", "so", ["{ten} để lại bao nhiêu công trình?", "Số công trình của {goi} là bao nhiêu?"]),
    "hoc_tro": ("so", "hoc_tro", ["{ten} đào tạo bao nhiêu học trò?", "{goi} có bao nhiêu học trò?"]),
}


def tao(n_doan, tien_to, nfd):
    rows = []
    for k in range(n_doan):
        f = nguoi(); van, co = doan(f); cid = f"{tien_to}c{k:04d}"
        loai = R.sample(list(HOI), 5)
        for j, lq in enumerate(loai):
            can, khoa, mau = HOI[lq]
            q = R.choice(mau).format(ten=f["ten"], goi=f["goi"])
            if can in co:
                ans = f[khoa]; bd, cau = co[can]
                st = bd + (cau.rfind(ans) if khoa == "hoc_tro" else cau.find(ans))   # tìm trong đúng câu chứa thông tin
            else:
                ans, st = "", -1
            rows.append(dict(id=f"{tien_to}q{k:04d}{j}", context_id=cid, context=van, question=q, answer=ans, answer_start=st))
    df = pd.DataFrame(rows)
    if nfd:
        m = [R.random() < 0.4 for _ in range(len(df))]
        df.loc[m, "context"] = df.loc[m, "context"].map(lambda s: unicodedata.normalize("NFD", s))
        df.loc[m, "question"] = df.loc[m, "question"].map(lambda s: unicodedata.normalize("NFD", s))
    return df


tr, te = tao(700, "t", False), tao(260, "e", True)
assert all(r.answer == "" or r.context[r.answer_start:r.answer_start + len(r.answer)] == r.answer for r in tr.itertuples())
tr.to_csv(D / "train.csv", index=False)
te[["id", "context_id", "context", "question"]].to_csv(D / "test.csv", index=False)
te[["id", "answer"]].sort_values("id").to_csv(A / "test.csv", index=False)
print(f"xong: train {len(tr)} câu hỏi ({(tr.answer == '').mean():.0%} không đáp án), test {len(te)}")
