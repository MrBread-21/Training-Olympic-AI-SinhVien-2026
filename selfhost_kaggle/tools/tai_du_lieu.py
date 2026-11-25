"""Tải dữ liệu gốc từ Hugging Face vào selfhost_kaggle/raw/ (hoặc thư mục SELFHOST_RAW).
    python tools/tai_du_lieu.py              # tất cả
    python tools/tai_du_lieu.py nordland vsfc

Tổng khoảng 24 GB, phần lớn là Nordland (2 mùa đông và hè, mỗi mùa khoảng 10-12 GB).
MVTec AD trên Hugging Face gồm hàng nghìn file nhỏ nên bị giới hạn tốc độ: chỉ tải 5 loại cần dùng, có thử lại.
"""
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from chung import RAW

DS = {  # tên: (repo, các file cần lấy; None = cả repo)
    "kvasir": ("kowndinya23/Kvasir-SEG", None),
    "aircraft_train": ("Multimodal-Fatima/FGVC_Aircraft_train", None),
    "aircraft_test": ("Multimodal-Fatima/FGVC_Aircraft_test", None),
    "voc": ("nateraw/pascal-voc-2012", None),
    "ucf": ("sayakpaul/ucf101-subset", ["UCF101_subset.tar.gz", "README.md"]),
    "vsfc": ("ura-hcmut/UIT-VSFC", None),
    "phoner": ("phucdev/PhoNER_COVID19", ["syllable/*", "README.md"]),
    "iwslt": ("thainq107/iwslt2015-en-vi", None),
    "viquad": ("taidng/UIT-ViQuAD2.0", None),
    "news": ("NamSyntax/vietnamese-news-classification", ["data/train-00000-of-00002.parquet", "README.md"]),
}
MVTEC_LOAI = {"bottle", "carpet", "hazelnut", "metal_nut", "screw"}


def tai(repo, f, dich):
    for lan in range(8):
        try:
            return hf_hub_download(repo, f, repo_type="dataset", local_dir=dich)
        except Exception as e:  # 429 (giới hạn tốc độ), 502...
            print("  thử lại sau", 30 * (lan + 1), "s:", repr(e)[:80], flush=True)
            time.sleep(30 * (lan + 1))
    raise RuntimeError(f"không tải được {repo}/{f}")


def nordland():
    d = RAW / "nordland"
    for f in ["README.md", "dataset_imageNames/nordland_imageNames.txt", "annotations/winter.csv", "annotations/summer.csv",
              "data/winter.tar.gz", "data/summer.tar.gz"]:
        tai("Somayeh-h/Nordland", f, d)
        print("  nordland", f, flush=True)


def mvtec():
    d = RAW / "mvtec"
    tai("Voxel51/mvtec-ad", "samples.json", d)
    s = json.load(open(d / "samples.json"))["samples"]
    files = [x["filepath"] for x in s if x["category"]["label"] in MVTEC_LOAI]
    for i, f in enumerate(files):
        if not (d / f).exists():
            tai("Voxel51/mvtec-ad", f, d)
        if i % 200 == 0:
            print(f"  mvtec {i}/{len(files)}", flush=True)


if __name__ == "__main__":
    ten = sys.argv[1:] or ["nordland", "mvtec", *DS]
    for k in ten:
        t = time.time()
        if k == "nordland":
            nordland()
        elif k == "mvtec":
            mvtec()
        else:
            repo, pats = DS[k]
            snapshot_download(repo, repo_type="dataset", local_dir=RAW / k, allow_patterns=pats, max_workers=4)
        print(k, "xong", round(time.time() - t), "s", flush=True)
