"""Nén <Bài>/upload + baseline.py + README.md (đề) thành <Bài>/<Bài>_data.zip để tải lên Kaggle (trang Data).
    python tools/dong_goi.py [Bài ...]      (không đối số = mọi bài đã dựng)"""
import sys
import zipfile

from chung import GOC

VAN_BAN = {".csv", ".json", ".jsonl", ".txt", ".md", ".py", ".ipynb"}

bai = sys.argv[1:] or sorted(p.name for p in GOC.iterdir() if (p / "upload").is_dir())
for b in bai:
    d = GOC / b
    z = d / f"{b}_data.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:  # ảnh, video, âm thanh: không nén thêm; văn bản: nén
        for p in sorted((d / "upload").rglob("*")):
            if p.is_file():
                f.write(p, p.relative_to(d / "upload"), zipfile.ZIP_DEFLATED if p.suffix in VAN_BAN else zipfile.ZIP_STORED)
        for extra in ["baseline.py", "baseline.ipynb", "README.md"]:
            if (d / extra).exists():
                f.write(d / extra, extra)
    print(f"{b}: {z.stat().st_size / 1e6:.0f} MB")
