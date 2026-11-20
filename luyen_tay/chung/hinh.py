"""Sinh ảnh hình học giả lập (64x64 RGB) cho các bài CV: nền có vân, một vật thể thuộc 6 lớp, kèm mặt nạ vật thể."""
import cv2
import numpy as np

LOP = ["tron", "vuong", "tam_giac", "chu_thap", "vanh", "sao"]


def nen(rng, S=64):
    """Nền: màu gốc + dải màu chuyển + nhiễu mịn."""
    base = rng.uniform(40, 200, 3)
    gx, gy = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1, 1, S))
    huong = rng.normal(size=2)
    grad = (gx * huong[0] + gy * huong[1])[..., None] * rng.uniform(10, 40, 3)
    noise = cv2.GaussianBlur(rng.normal(0, 25, (S, S, 3)).astype(np.float32), (0, 0), rng.uniform(1, 3))
    return base + grad + noise


def da_giac(cx, cy, r, n, goc, sao=False):
    k = 2 * n if sao else n
    a = goc + np.arange(k) * 2 * np.pi / k
    rr = np.where(np.arange(k) % 2 == 0, r, r * 0.45) if sao else np.full(k, r)
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1).astype(np.int32)


def ve(lop, rng, S=64, r=None, tam=None, bg=None, mau=None):
    """Trả (ảnh uint8 HxWx3, mặt nạ uint8 HxW) cho lớp `lop` (chỉ số hoặc tên).
    bg, mau: nền và màu vật thể cố định (để nhiều ảnh cùng một "phiên chụp")."""
    lop = LOP.index(lop) if isinstance(lop, str) else lop
    img = nen(rng, S) if bg is None else bg + rng.normal(0, 6, bg.shape)
    m = np.zeros((S, S), np.uint8)
    r = r or rng.uniform(11, 20)
    cx, cy = tam or rng.uniform(r + 2, S - r - 2, 2)
    goc = rng.uniform(0, 2 * np.pi)
    c = (int(cx), int(cy))
    if lop == 0:
        cv2.circle(m, c, int(r), 1, -1)
    elif lop == 1:
        cv2.fillPoly(m, [da_giac(cx, cy, r, 4, goc)], 1)
    elif lop == 2:
        cv2.fillPoly(m, [da_giac(cx, cy, r, 3, goc)], 1)
    elif lop == 3:
        w = max(3, int(r * 0.35))
        cv2.line(m, (int(cx - r), int(cy)), (int(cx + r), int(cy)), 1, w)
        cv2.line(m, (int(cx), int(cy - r)), (int(cx), int(cy + r)), 1, w)
        M = cv2.getRotationMatrix2D(c, np.degrees(goc), 1.0)
        m = cv2.warpAffine(m, M, (S, S), flags=cv2.INTER_NEAREST)
    elif lop == 4:
        cv2.circle(m, c, int(r), 1, max(3, int(r * 0.3)))
    else:
        cv2.fillPoly(m, [da_giac(cx, cy, r, 5, goc, sao=True)], 1)
    if mau is None:
        mau = rng.uniform(0, 255, 3)
        while np.abs(mau - img.mean((0, 1))).sum() < 150:          # vật thể khác màu nền rõ ràng
            mau = rng.uniform(0, 255, 3)
    img = np.where(m[..., None] > 0, mau + rng.normal(0, 8, (S, S, 3)), img)
    return np.clip(img, 0, 255).astype(np.uint8), m


def lam_hong(img, rng, manh=1.0):
    """Biến đổi kiểu ảnh test thực tế: xoay 90/lật, mờ, nhiễu, lệch màu, JPEG."""
    img = np.rot90(img, rng.integers(4)).copy()
    if rng.random() < 0.5: img = img[:, ::-1].copy()
    x = img.astype(np.float32)
    if rng.random() < 0.6 * manh: x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 0.3 + 1.2 * manh))
    if rng.random() < 0.6 * manh: x = x + rng.normal(0, rng.uniform(3, 3 + 22 * manh), x.shape)
    if rng.random() < 0.6 * manh: x = x * rng.uniform(0.75, 1.25, 3) + rng.uniform(-25, 25, 3)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.6 * manh:
        _, e = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(10, 50))])
        x = cv2.imdecode(e, cv2.IMREAD_UNCHANGED)
    return x


def luu(path, img):
    cv2.imwrite(str(path), img[..., ::-1])                         # RGB sang BGR cho cv2
