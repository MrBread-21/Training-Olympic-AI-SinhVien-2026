"""Tạo dữ liệu bài 02 (ảnh hình học 64x64 + 5 câu mô tả tiếng Việt mỗi ảnh; CPU khoảng 10 giây).
Mỗi ảnh có 1-2 vật: hình (tròn, vuông, tam giác, sao), màu (đỏ, xanh lá, xanh dương, vàng, tím, trắng), cỡ (nhỏ, lớn),
vị trí (bên trái, bên phải, phía trên, phía dưới, ở giữa). Ảnh có nhiễu và nền đổi màu.
Train 3.000 ảnh, test 600 ảnh. Số tổ hợp rất lớn nên ảnh test hiếm khi trùng hẳn một ảnh train."""
import json
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
D, A = HERE / "data", HERE / ".dap_an"
for p in (D / "train", D / "test", A):
    p.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(202); R = random.Random(202)
HINH = ["tròn", "vuông", "tam giác", "sao"]
MAU = {"đỏ": (220, 40, 40), "xanh lá": (40, 180, 60), "xanh dương": (40, 80, 220), "vàng": (230, 210, 40), "tím": (150, 60, 190), "trắng": (245, 245, 245)}
VT = {"bên trái": (16, 32), "bên phải": (48, 32), "phía trên": (32, 16), "phía dưới": (32, 48), "ở giữa": (32, 32)}


def ve_vat(img, hinh, mau, co, vt):
    cx, cy = (v + int(rng.integers(-3, 4)) for v in VT[vt]); r = 9 if co == "lớn" else 5
    c = tuple(int(np.clip(v + rng.integers(-20, 21), 0, 255)) for v in MAU[mau])
    if hinh == "tròn":
        cv2.circle(img, (cx, cy), r, c, -1)
    elif hinh == "vuông":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), c, -1)
    elif hinh == "tam giác":
        cv2.fillPoly(img, [np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)], c)
    else:
        a = np.pi / 2 + np.arange(10) * np.pi / 5; rr = np.where(np.arange(10) % 2 == 0, r + 1, r * 0.45)
        cv2.fillPoly(img, [np.stack([cx - rr * np.cos(a), cy - rr * np.sin(a)], 1).astype(np.int32)], c)


def mo_ta(vat):
    if len(vat) == 1:
        h, m, c, v = vat[0]
        return [f"một hình {h} màu {m} {v}", f"có một hình {h} {m} nằm {v} bức ảnh", f"hình {h} {c} màu {m}",
                f"bức ảnh có một hình {h} màu {m}", f"một hình {h} {c} màu {m} {v}"]
    (h1, m1, c1, v1), (h2, m2, c2, v2) = vat
    return [f"một hình {h1} màu {m1} và một hình {h2} màu {m2}", f"hình {h1} {m1} {v1} và hình {h2} {m2} {v2}",
            f"bức ảnh có hai hình là hình {h1} và hình {h2}", f"có một hình {h1} {c1} màu {m1} và một hình {h2} {c2} màu {m2}",
            f"hai hình gồm hình {h1} màu {m1} và hình {h2} màu {m2}"]


def tao(n, thu_muc, tien_to):
    caps = {}
    for i in range(n):
        nen = rng.uniform(20, 110, 3)
        img = (np.ones((64, 64, 3)) * nen + rng.normal(0, 6, (64, 64, 3))).clip(0, 255).astype(np.uint8)
        so = 1 if R.random() < 0.45 else 2
        vts = R.sample(list(VT), so)
        vat = [(R.choice(HINH), R.choice(list(MAU)), R.choice(["nhỏ", "lớn"]), v) for v in vts]
        if so == 2 and vat[0][:2] == vat[1][:2]:
            vat[1] = (R.choice([h for h in HINH if h != vat[0][0]]),) + vat[1][1:]
        for h, m, c, v in vat:
            ve_vat(img, h, m, c, v)
        img = np.clip(img.astype(np.float32) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
        k = f"{tien_to}{i:05d}"
        cv2.imwrite(str(thu_muc / f"{k}.png"), img[..., ::-1])
        caps[k] = mo_ta(vat)
    return caps


tr, te = tao(3000, D / "train", "tr"), tao(600, D / "test", "te")
(D / "train_captions.json").write_text(json.dumps(tr, ensure_ascii=False, indent=0), encoding="utf-8")
(A / "test_captions.json").write_text(json.dumps(te, ensure_ascii=False, indent=0), encoding="utf-8")
print(f"xong: train {len(tr)} ảnh, test {len(te)} ảnh, 5 câu mỗi ảnh")
