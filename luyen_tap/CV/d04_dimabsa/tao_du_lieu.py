"""Tạo dữ liệu bài 04 (review tiếng Việt giả lập, miền ứng dụng giao đồ ăn, đúng 10 category của đề OlpAI'26; CPU vài giây).
Mỗi câu 1-3 bộ tứ (aspect, category, opinion, valence, arousal). VA của opinion phụ thuộc từ gốc và từ bổ nghĩa:
"rất / quá / siêu / xỉu" đẩy xa trung tính và tăng arousal, "hơi / khá" kéo về trung tính, "không / chẳng" đảo chiều.
Bẫy: teencode (ko, k, qá, wá, dc, j), khoảng 30% aspect của test là từ không có trong train, aspect ẩn "NULL".
A (hồi quy VA khi biết aspect): train 2500 câu, test 800. B (trích bộ tứ): train 2500 câu, test 800 (câu khác A)."""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = random.Random(404)
D, A = HERE / "data", HERE / ".dap_an"
D.mkdir(exist_ok=True); A.mkdir(exist_ok=True)

ASP = {  # category: (aspect dùng trong train, aspect chỉ có ở test)
    "APP#PERFORMANCE": (["app", "ứng dụng", "app đặt món", "tốc độ app"], ["phần mềm", "bản cập nhật mới"]),
    "APP#UI_UX": (["giao diện", "thiết kế app", "menu"], ["nút thanh toán", "màn hình đặt món"]),
    "STAFF#ATTITUDE": (["shipper", "tài xế", "anh shipper"], ["chị giao hàng", "bạn tài xế"]),
    "STAFF#PROFESSIONALISM": (["tổng đài", "nhân viên tư vấn", "bộ phận chăm sóc khách hàng"], ["nhân viên hỗ trợ"]),
    "PRODUCT#QUALITY": (["món ăn", "đồ ăn", "trà sữa", "phở", "bánh mì", "cơm tấm", "pizza"], ["bún bò", "gà rán", "chè thái", "hủ tiếu"]),
    "PRODUCT#DESIGN": (["bao bì", "hộp đựng", "cách đóng gói"], ["túi giữ nhiệt", "ly nhựa"]),
    "SERVICE#TIME": (["thời gian giao", "tốc độ giao hàng", "giao hàng"], ["thời gian chờ"]),
    "SERVICE#GENERAL": (["dịch vụ", "chất lượng phục vụ", "phục vụ"], ["dịch vụ hậu mãi"]),
    "PRICE#RATES": (["giá", "phí ship", "giá cả"], ["phí dịch vụ", "giá món"]),
    "PRICE#PROMO": (["khuyến mãi", "mã giảm giá", "voucher"], ["chương trình tích điểm"]),
}
OP = {  # category: [(opinion gốc, V, A)]
    "APP#PERFORMANCE": [("mượt", 7.4, 6.0), ("lag", 2.4, 7.2), ("nhanh", 7.2, 6.6), ("hay bị đơ", 2.2, 7.4), ("ổn định", 6.8, 4.8)],
    "APP#UI_UX": [("dễ dùng", 7.3, 5.6), ("rối", 3.0, 6.0), ("đẹp", 7.5, 6.0), ("khó hiểu", 3.1, 6.2), ("gọn gàng", 7.0, 5.2)],
    "STAFF#ATTITUDE": [("dễ thương", 7.9, 6.4), ("thô lỗ", 1.8, 7.9), ("nhiệt tình", 7.8, 6.8), ("cáu gắt", 2.0, 7.7), ("lịch sự", 7.2, 5.4)],
    "STAFF#PROFESSIONALISM": [("chuyên nghiệp", 7.4, 5.8), ("thiếu chuyên nghiệp", 2.6, 6.6), ("tận tình", 7.7, 6.2), ("hời hợt", 3.0, 5.6)],
    "PRODUCT#QUALITY": [("ngon", 7.8, 6.8), ("dở", 2.4, 6.4), ("nguội", 3.2, 5.4), ("tươi", 7.2, 6.0), ("nhạt", 3.4, 5.0), ("đậm đà", 7.6, 6.6)],
    "PRODUCT#DESIGN": [("chắc chắn", 7.0, 5.4), ("móp méo", 2.8, 6.2), ("xinh xắn", 7.6, 6.4), ("bị rò rỉ", 2.2, 7.0)],
    "SERVICE#TIME": [("nhanh", 7.4, 6.8), ("chậm", 2.8, 6.8), ("đúng giờ", 7.0, 5.4), ("trễ", 2.6, 7.0)],
    "SERVICE#GENERAL": [("tốt", 7.2, 5.8), ("tệ", 2.0, 7.2), ("chu đáo", 7.6, 6.0), ("kém", 2.6, 6.4)],
    "PRICE#RATES": [("rẻ", 7.2, 6.2), ("đắt", 2.8, 6.6), ("hợp lý", 6.8, 5.0), ("chát", 2.5, 7.0)],
    "PRICE#PROMO": [("hấp dẫn", 7.6, 7.0), ("ít", 3.4, 5.2), ("xịn", 7.8, 7.2), ("khó dùng", 3.0, 6.0)],
}
NULL_OP = [("SERVICE#GENERAL", "sẽ quay lại ủng hộ", 7.6, 6.2), ("PRICE#RATES", "đáng đồng tiền", 7.4, 6.0),
           ("SERVICE#TIME", "chờ mòn mỏi", 2.4, 6.8), ("SERVICE#GENERAL", "không bao giờ đặt nữa", 1.8, 7.6)]
TANG = [("rất", "trước"), ("quá", "trước"), ("siêu", "trước"), ("cực kỳ", "trước"), ("xỉu", "sau"), ("dã man", "sau")]
GIAM = ["hơi", "khá"]
PHU = ["không", "chẳng"]
TEEN = {"không": ["ko", "k"], "quá": ["qá", "wá"], "rất": ["rất"], "được": ["dc", "đc"]}
NOI = [", nhưng ", " mà ", ", còn ", " và ", ", "]
DAU = ["", "", "Lần đầu đặt thử, ", "Nói chung là ", "Thật sự ", "Đặt hôm qua, ", "Review nhanh: "]


def opinion(cat):
    goc, v, a = R.choice(OP[cat])
    r = R.random()
    if r < 0.25:                                                   # tăng cường
        tu, vt = R.choice(TANG)
        txt = f"{tu} {goc}" if vt == "trước" else f"{goc} {tu}"
        v, a = 5 + (v - 5) * 1.35, a + 0.9
    elif r < 0.4:                                                  # giảm nhẹ
        txt = f"{R.choice(GIAM)} {goc}"; v, a = 5 + (v - 5) * 0.55, a - 0.8
    elif r < 0.52 and " " not in goc:                              # phủ định
        txt = f"{R.choice(PHU)} {goc}"; v, a = 5 - (v - 5) * 0.55, a - 0.4
    else:
        txt = goc
    v, a = v + R.gauss(0, 0.25), a + R.gauss(0, 0.25)
    return txt, round(min(9, max(1, v)), 2), round(min(9, max(1, a)), 2)


def teen(s):
    out = []
    for w in s.split(" "):
        out.append(R.choice(TEEN[w]) if w in TEEN and R.random() < 0.4 else w)
    return " ".join(out)


def cau(test):
    n = R.choices([1, 2, 3], [0.45, 0.4, 0.15])[0]
    cats = R.sample(list(ASP), n)
    phan, quads = [], []
    for k, cat in enumerate(cats):
        if R.random() < 0.08:
            c, op, v, a = R.choice(NULL_OP)
            op = teen(op); phan.append(op); quads.append(("NULL", c, op, v, a)); continue
        tr, moi = ASP[cat]
        asp = R.choice(moi) if test and R.random() < 0.3 else R.choice(tr)
        op, v, a = opinion(cat); op = teen(op)
        mau = R.choice(["{a} {o}", "{a} thì {o}", "{a} {o} thật", "{a} của quán {o}", "{o} cái {a}"] if k else ["{a} {o}", "{a} thì {o}", "{a} {o} thật"])
        phan.append(mau.format(a=asp, o=op)); quads.append((asp, cat, op, v, a))
    s = phan[0]
    for p in phan[1:]:
        s += R.choice(NOI) + p
    s = R.choice(DAU) + s
    s = s[0].upper() + s[1:] + R.choice([".", "!", " .", "..."])
    # aspect / opinion phải là chuỗi con đúng của câu: sửa chữ hoa đầu câu nếu bị viết hoa
    quads = [(q[0] if q[0] == "NULL" or q[0] in s else q[0][0].upper() + q[0][1:],) + q[1:] for q in quads]
    quads = [q[:2] + ((q[2] if q[2] in s else q[2][0].upper() + q[2][1:]),) + q[3:] for q in quads]
    return s, quads


def tao(n, test, tien_to):
    out = []
    for i in range(n):
        s, q = cau(test)
        out.append({"id": f"{tien_to}{i:04d}", "text": s,
                    "quadruplets": [dict(aspect=a, category=c, opinion=o, valence=v, arousal=ar) for a, c, o, v, ar in q]})
    return out


ghi = lambda p, rows: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
# A: hồi quy VA cho aspect cho trước (bỏ aspect NULL, mỗi câu có ít nhất một aspect)
a_tr = [r for r in tao(2900, False, "atr") if any(q["aspect"] != "NULL" for q in r["quadruplets"])][:2500]
a_te = [r for r in tao(1000, True, "ate") if any(q["aspect"] != "NULL" for q in r["quadruplets"])][:800]
nhan_a = lambda r: [dict(aspect=q["aspect"], valence=q["valence"], arousal=q["arousal"]) for q in r["quadruplets"] if q["aspect"] != "NULL"]
ghi(D / "a_train.jsonl", [dict(id=r["id"], text=r["text"], aspects=nhan_a(r)) for r in a_tr])
ghi(D / "a_test.jsonl", [dict(id=r["id"], text=r["text"], aspects=[{"term": x["aspect"]} for x in nhan_a(r)]) for r in a_te])
ghi(A / "a_test.jsonl", [dict(id=r["id"], aspects=nhan_a(r)) for r in a_te])
# B: trích bộ tứ
b_tr, b_te = tao(2500, False, "btr"), tao(800, True, "bte")
ghi(D / "b_train.jsonl", b_tr)
ghi(D / "b_test.jsonl", [dict(id=r["id"], text=r["text"]) for r in b_te])
ghi(A / "b_test.jsonl", b_te)
print("xong: A 2500 / 800 câu, B 2500 / 800 câu")
