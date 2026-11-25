# %% [markdown]
# # Baseline - Nhận dạng từ khoá trong âm thanh (Speech Commands)
#
# Không học sâu: log-mel spectrogram (40 dải, tự tính bằng numpy), lấy trung bình và độ lệch chuẩn theo thời gian
# của từng dải, rồi Logistic Regression có cân bằng lớp. Tự chấm macro-F1 trên 20% người nói của train để riêng,
# rồi ghi `submission.csv` (`id,label`).
#
# Chạy: `python baseline.py [thư_mục_dữ_liệu]`. Trên Kaggle tự tìm dữ liệu trong `/kaggle/input`.
# Muốn dự đoán cả private test: giải nén `private_test.zip` (mật khẩu công bố ở tiếng cuối) vào thư mục làm việc.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    D = Path(sys.argv[1])
else:
    tim = sorted(Path("/kaggle/input").rglob("train.csv")) if Path("/kaggle/input").exists() else []
    D = tim[0].parent if tim else Path(".")
print("dữ liệu:", D)


def mel_fb(n_mel=40, n_fft=512, sr=16000):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mel = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mel + 2)
    bins = np.floor((n_fft + 1) * hz(mel) / sr).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1), np.float32)
    for i in range(n_mel):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        fb[i, a:b] = (np.arange(a, b) - a) / max(1, b - a)
        fb[i, b:c] = (c - np.arange(b, c)) / max(1, c - b)
    return fb


FB, CUA = mel_fb(), np.hanning(400).astype(np.float32)


def dac_trung(p):
    sr, a = wavfile.read(p)
    a = a.astype(np.float32) / 32768
    a = np.pad(a, (0, max(0, 16000 - len(a))))[:16000]
    khung = np.lib.stride_tricks.sliding_window_view(a, 400)[::160] * CUA  # 25 ms, bước 10 ms
    S = np.log(FB @ (np.abs(np.fft.rfft(khung, 512)) ** 2).T + 1e-6)  # (40, số khung)
    return np.concatenate([S.mean(1), S.std(1), np.diff(S, axis=1).std(1)])


tr = pd.read_csv(D / "train.csv")
X = np.stack([dac_trung(D / "train" / f"{i}.wav") for i in tr.id])
print("train", X.shape, tr.label.value_counts().to_dict())


def mo_hinh():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))


# %% [markdown]
# ## Tự chấm: để riêng 20% người nói của train

# %%
rng = np.random.default_rng(42)
loa = tr.speaker.unique(); val = tr.speaker.isin(rng.choice(loa, len(loa) // 5, replace=False)).values
m = mo_hinh().fit(X[~val], tr.label[~val])
print(f"val macro-F1 (âm thanh sạch) = {f1_score(tr.label[val], m.predict(X[val]), average='macro'):.4f}")

# %% [markdown]
# ## Train lại trên toàn bộ train, dự đoán test (public) và private_test (nếu đã giải nén)

# %%
m = mo_hinh().fit(X, tr.label)
tep = sorted((D / "test").glob("*.wav"))
for thu in [Path("private_test"), D / "private_test"]:
    if thu.is_dir():
        tep += sorted(thu.glob("*.wav")); break
pd.DataFrame({"id": [t.stem for t in tep], "label": m.predict(np.stack([dac_trung(t) for t in tep]))}).to_csv("submission.csv", index=False)
print(f"ghi submission.csv: {len(tep)} đoạn âm thanh")
