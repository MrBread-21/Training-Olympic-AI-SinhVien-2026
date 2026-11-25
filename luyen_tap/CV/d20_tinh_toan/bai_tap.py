"""Cấp 1 - Tính toán cơ bản bằng numpy.

Viết thân từng hàm (thay dòng `raise NotImplementedError`). Không dùng sklearn / scipy / torch trong file này,
chỉ numpy và thư viện chuẩn. Hạn chế vòng lặp Python khi numpy làm được (broadcasting, axis, fancy indexing).

Chấm:  python kiem_tra.py          (mọi bài)
       python kiem_tra.py 9        (chỉ bài 9)
"""
import re
import numpy as np


# ---------------------------------------------------------------- A. Chuẩn hoá dữ liệu

def bai01_mean_std_kenh(X):
    """X: mảng uint8 (N, C, H, W). Trả (mean, std), mỗi cái là mảng float (C,), tính trên mọi ảnh và mọi pixel
    của từng kênh. std là độ lệch chuẩn tổng thể (chia N*H*W, không trừ 1)."""
    raise NotImplementedError


def bai02_chuan_hoa(X, mean, std):
    """X: (N, C, H, W); mean, std: (C,). Trả mảng float32 cùng shape, (X - mean) / std theo từng kênh."""
    raise NotImplementedError


def bai03_one_hot(y, k):
    """y: mảng int (N,) giá trị 0..k-1. Trả mảng float32 (N, k), hàng i có số 1 ở cột y[i]."""
    raise NotImplementedError


# ---------------------------------------------------------------- B. Softmax và loss

def bai04_softmax(z):
    """z: (N, K). Softmax theo từng hàng. Phải ổn định số học: z có thể chứa giá trị cỡ 1000 mà không ra nan/inf."""
    raise NotImplementedError


def bai05_cross_entropy(logits, y):
    """logits: (N, K), y: (N,) nhãn int. Trả cross-entropy trung bình (số thực): mean_i(-log softmax(logits_i)[y_i]).
    Ổn định số học (dùng log-sum-exp, không lấy log của softmax đã tính)."""
    raise NotImplementedError


# ---------------------------------------------------------------- C. Metric phân loại

def bai06_accuracy(y, p):
    """y, p: (N,) nhãn thật và nhãn dự đoán. Trả tỉ lệ đúng."""
    raise NotImplementedError


def bai07_ma_tran_nham(y, p, k):
    """Trả ma trận int (k, k): phần tử [i, j] = số mẫu có nhãn thật i và dự đoán j. Không dùng vòng lặp qua mẫu."""
    raise NotImplementedError


def bai08_macro_f1(y, p, k):
    """Macro-F1 trên k lớp 0..k-1: F1 từng lớp rồi trung bình đều. Lớp nào có TP + FP + FN = 0 thì F1 lớp đó là 0."""
    raise NotImplementedError


def bai09_auroc(y, s):
    """y: (N,) nhãn 0/1, s: (N,) điểm (càng cao càng có khả năng là 1). Trả AUROC.
    Dùng công thức hạng (Mann-Whitney): AUROC = (tổng hạng của mẫu dương - n1(n1+1)/2) / (n1 * n0),
    hạng tính từ 1 theo s tăng dần, các điểm bằng nhau nhận hạng trung bình."""
    raise NotImplementedError


# ---------------------------------------------------------------- D. Khoảng cách và láng giềng

def bai10_cosine(A, B):
    """A: (n, d), B: (m, d). Trả ma trận (n, m) độ tương đồng cosine giữa từng hàng của A và từng hàng của B."""
    raise NotImplementedError


def bai11_khoang_cach(A, B):
    """A: (n, d), B: (m, d). Trả ma trận (n, m) khoảng cách Euclid, không dùng vòng lặp.
    Gợi ý: |a - b|^2 = |a|^2 + |b|^2 - 2 a.b; nhớ chặn giá trị âm nhỏ do sai số trước khi lấy căn."""
    raise NotImplementedError


def bai12_knn(Xtr, ytr, Xte, k):
    """Phân loại kNN: mỗi mẫu của Xte lấy k mẫu train gần nhất (Euclid), bỏ phiếu đa số.
    Hoà phiếu thì chọn nhãn nhỏ nhất. Trả (len(Xte),) nhãn int."""
    raise NotImplementedError


# ---------------------------------------------------------------- E. Chuỗi thời gian và đoạn

def bai13_lam_muot(x, w):
    """x: (T,), w: cửa sổ lẻ. Trung bình trượt cùng độ dài T; hai đầu đệm bằng giá trị biên (mode 'edge')."""
    raise NotImplementedError


def bai14_tiou(a, b):
    """a = (s1, e1), b = (s2, e2): đoạn khung hình, tính cả hai đầu (khung 3..5 có 3 khung).
    Trả tIoU = số khung giao / số khung hợp. Không giao thì 0."""
    raise NotImplementedError


def bai15_jaccard_tu(a, b):
    """a, b: chuỗi. Chữ thường, thay mọi ký tự không phải chữ/số/khoảng trắng bằng khoảng trắng, tách theo khoảng trắng,
    rồi Jaccard của hai tập từ. Cả hai tập rỗng thì trả 1.0."""
    raise NotImplementedError


def bai16_doan_tong_lon_nhat(x):
    """x: (T,) số thực. Trả (i, j): đoạn liên tục x[i..j] (tính cả j) có tổng lớn nhất. Yêu cầu O(T) (thuật toán Kadane)."""
    raise NotImplementedError
