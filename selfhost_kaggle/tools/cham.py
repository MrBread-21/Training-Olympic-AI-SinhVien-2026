"""Chấm tại chỗ (dự phòng khi Kaggle lỗi), dùng đúng <Bài>/host/metric.py đã đưa lên Kaggle.
    python tools/cham.py Kvasir_PolypSegmentation submission.csv
In điểm Public, Private và toàn bộ."""
import importlib.util
import sys

import pandas as pd

from chung import GOC

bai, nop = sys.argv[1], sys.argv[2]
host = GOC / bai / "host"
spec = importlib.util.spec_from_file_location("metric_bai", host / "metric.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sol = pd.read_csv(host / "solution.csv", dtype=str, keep_default_na=False)
sub = pd.read_csv(nop, dtype=str, keep_default_na=False)
for ten, part in [("Public", sol[sol.Usage == "Public"]), ("Private", sol[sol.Usage == "Private"]), ("Toàn bộ", sol)]:
    print(f"{ten:8s} {mod.score(part.reset_index(drop=True), sub, 'id'):.4f}")
