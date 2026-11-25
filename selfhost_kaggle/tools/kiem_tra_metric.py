"""Kiểm tra metric của từng bài sau khi dựng đề:
- nộp chính đáp án (bỏ Usage, đổi tên cột cho đúng định dạng nộp) phải ra điểm tối đa;
- nộp file mẫu phải chấm được;
- nộp thiếu dòng phải báo ParticipantVisibleError (thí sinh thấy được lỗi).
Ghi kèm <Bài>/host/nop_dung_het.csv để nộp thử trên Kaggle (Sandbox Submission phải ra điểm tối đa).
    python tools/kiem_tra_metric.py Nordland_PlaceRecognition VSFC_Sentiment ...
"""
import importlib.util
import sys

import pandas as pd

from chung import GOC

DOI_TEN = {"positives": "predictions", "passage_id": "passage_ids"}
for b in sys.argv[1:]:
    host = GOC / b / "host"
    spec = importlib.util.spec_from_file_location("metric_bai", host / "metric.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sol = pd.read_csv(host / "solution.csv", dtype=str, keep_default_na=False)
    dung = sol.drop(columns=[c for c in ("Usage", "height", "width") if c in sol]).rename(columns=DOI_TEN)
    if "label" in dung and b.startswith("MVTec"):
        dung = dung.rename(columns={"label": "score"})  # AUROC: nộp cột score
    dung.to_csv(host / "nop_dung_het.csv", index=False)
    mau = pd.read_csv(GOC / b / "upload/sample_submission.csv", dtype=str, keep_default_na=False)
    pub = sol[sol.Usage == "Public"].reset_index(drop=True)
    print(f"{b:28s} đúng hết: {mod.score(pub, dung, 'id'):.4f}   file mẫu: {mod.score(pub, mau, 'id'):.4f}")
    try:
        mod.score(pub, dung.iloc[:-5], "id")
        print("   LỖI: thiếu dòng mà không báo")
    except mod.ParticipantVisibleError as e:
        print("   thiếu dòng báo lỗi:", e)
