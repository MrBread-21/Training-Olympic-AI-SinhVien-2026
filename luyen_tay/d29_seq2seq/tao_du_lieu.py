"""Tạo dữ liệu bài 29: dịch Anh sang Việt theo một ngữ pháp nhỏ: đổi trật tự tính từ, lượng từ, số đếm 1-99
(hai mươi mốt, mười lăm...), "của tôi", thì, tân ngữ kép nối bằng "and". 12000 cặp train, 500 val, 1000 test. CPU vài giây."""
import random
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = random.Random(2901)
D, A = HERE / "data", HERE / ".dap_an"
D.mkdir(exist_ok=True); A.mkdir(exist_ok=True)
sp = lambda s: dict(x.split(":") for x in s.split("|"))
NGUOI = sp("cat:con mèo|dog:con chó|teacher:cô giáo|student:học sinh|farmer:bác nông dân|doctor:bác sĩ|child:đứa trẻ|bird:con chim|"
           "girl:cô bé|boy:cậu bé|king:nhà vua|queen:hoàng hậu|singer:ca sĩ|driver:tài xế|nurse:y tá|cook:đầu bếp|baker:thợ làm bánh|"
           "painter:họa sĩ|monkey:con khỉ|horse:con ngựa|tiger:con hổ|fisherman:ngư dân|soldier:người lính|grandmother:bà ngoại|uncle:chú")
VAT = {k: tuple(v.split(" ", 1)) for k, v in sp(
    "book:quyển sách|apple:quả táo|car:chiếc xe|house:ngôi nhà|letter:lá thư|picture:bức tranh|ball:quả bóng|cake:chiếc bánh|"
    "flower:bông hoa|chair:cái ghế|table:cái bàn|hat:cái mũ|shirt:cái áo|shoe:chiếc giày|bag:cái túi|cup:cái cốc|bottle:cái chai|"
    "orange:quả cam|banana:quả chuối|fish:con cá|egg:quả trứng|pen:cây bút|lamp:cái đèn|boat:chiếc thuyền|bicycle:chiếc xe đạp|"
    "clock:cái đồng hồ|mirror:cái gương|kite:con diều|key:chiếc chìa khóa|ring:chiếc nhẫn").items()}
TT = sp("red:đỏ|big:to|small:nhỏ|old:cũ|new:mới|beautiful:đẹp|green:xanh|white:trắng|cheap:rẻ|expensive:đắt|black:đen|yellow:vàng|"
        "heavy:nặng|light:nhẹ|long:dài|short:ngắn|clean:sạch|dirty:bẩn|strange:lạ|famous:nổi tiếng")
DT = sp("bought:đã mua|saw:đã thấy|painted:đã vẽ|sold:đã bán|found:đã tìm thấy|made:đã làm|lost:đã làm mất|washed:đã rửa|"
        "carried:đã mang|broke:đã làm vỡ|likes:thích|wants:muốn|needs:cần|hates:ghét|keeps:giữ|will buy:sẽ mua|will sell:sẽ bán|"
        "will paint:sẽ vẽ|will find:sẽ tìm|will carry:sẽ mang")
TG = sp("yesterday:hôm qua|today:hôm nay|last week:tuần trước|in the morning:vào buổi sáng|every day:mỗi ngày|tomorrow:ngày mai|"
        "last year:năm ngoái|at night:vào ban đêm|this afternoon:chiều nay")
EN1 = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
EN10 = "twenty thirty forty fifty sixty seventy eighty ninety".split()
VI1 = "một hai ba bốn năm sáu bảy tám chín".split()


def so(n):
    """(tiếng Anh, tiếng Việt) của số 1-99."""
    en = EN1[n - 1] if n < 20 else EN10[n // 10 - 2] + ("" if n % 10 == 0 else " " + EN1[n % 10 - 1])
    if n < 10: vi = VI1[n - 1]
    elif n < 20: vi = "mười" + ("" if n == 10 else " " + ("lăm" if n == 15 else VI1[n - 11]))
    else:
        u = n % 10
        vi = f"{VI1[n // 10 - 1]} mươi" + ("" if u == 0 else " " + {1: "mốt", 5: "lăm"}.get(u, VI1[u - 1]))
    return en, vi


def tan_ngu():
    o, n = R.choice(list(VAT)), R.choice([R.randint(1, 9), R.randint(1, 99)])
    a = R.choice(list(TT)) if R.random() < 0.6 else None
    ne, nv = so(n)
    return [ne] + ([a] if a else []) + [o + ("s" if n > 1 and not o.endswith("s") and o != "fish" else "")], [nv, *VAT[o]] + ([TT[a]] if a else [])


def cau():
    n, my = R.choice(list(NGUOI)), R.random() < 0.3
    a1 = R.choice(list(TT)) if R.random() < 0.5 else None
    en = ["my" if my else "the"] + ([a1] if a1 else []) + [n]
    vi = NGUOI[n].split() + ([TT[a1]] if a1 else []) + (["của", "tôi"] if my else [])
    v = R.choice(list(DT)); en += v.split(); vi += DT[v].split()
    e, w = tan_ngu(); en += e; vi += w
    if R.random() < 0.3:
        e, w = tan_ngu(); en += ["and"] + e; vi += ["và"] + w
    if R.random() < 0.6:
        t = R.choice(list(TG)); en += t.split(); vi += TG[t].split()
    return " ".join(en), " ".join(vi)


seen, rows = set(), []
while len(rows) < 13500:
    e, v = cau()
    if e not in seen: seen.add(e); rows.append((e, v))
df = pd.DataFrame(rows, columns=["en", "vi"])
df[:12000].to_csv(D / "train.csv", index=False)
df[12000:12500].to_csv(D / "val.csv", index=False)
te = df[12500:].reset_index(drop=True).assign(id=lambda d: [f"s{i:04d}" for i in range(len(d))])
te[["id", "en"]].to_csv(D / "test.csv", index=False)
te[["id", "vi"]].to_csv(A / "test.csv", index=False)
print("xong: 12000 train, 500 val,", len(te), "test | ví dụ:", rows[0], "|", so(21), so(15), so(45))
