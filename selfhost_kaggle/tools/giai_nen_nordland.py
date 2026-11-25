"""Giải nén 1 mùa Nordland, chỉ giữ 27.592 khung trong danh sách lọc chuẩn (bỏ khung tàu chạy dưới 15 km/h
và khung trong đường hầm), lưu JPEG 640x360 như đề OlpAI'26.  Mỗi mùa khoảng 10-15 phút.
    python tools/giai_nen_nordland.py winter
    python tools/giai_nen_nordland.py summer
"""
import io
import sys
import tarfile
import time

from PIL import Image

from chung import RAW

mua = sys.argv[1]
d = RAW / "nordland"
giu = set(open(d / "dataset_imageNames/nordland_imageNames.txt").read().split())
out = d / "anh640" / mua
out.mkdir(parents=True, exist_ok=True)
t0, n = time.time(), 0
with tarfile.open(d / f"data/{mua}.tar.gz", "r|gz") as t:  # đọc tuần tự, không giải nén cả file ra đĩa
    for m in t:
        ten = m.name.split("/")[-1]
        if not m.isfile() or ten not in giu:
            continue
        im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
        im.save(out / ten.replace(".png", ".jpg"), quality=90)
        n += 1
        if n % 5000 == 0:
            print(mua, n, round(time.time() - t0), "s", flush=True)
print(mua, "xong", n, "ảnh", round(time.time() - t0), "s")
