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
    cot_id = "query_id" if "query_id" in sol.columns else "id"
    if cot_id == "query_id":  # Nordland: nộp một winter_id đúng cho mỗi query_id
        dung = pd.DataFrame({"query_id": sol.query_id, "winter_id": sol.positives.str.split().str[0]})
    dung.to_csv(host / "nop_dung_het.csv", index=False)
    mau = pd.read_csv(GOC / b / "upload/sample_submission.csv", dtype=str, keep_default_na=False)
    pub = sol[sol.Usage == "Public"].reset_index(drop=True)
    print(f"{b:28s} đúng hết: {mod.score(pub, dung, cot_id):.4f}   file mẫu: {mod.score(pub, mau, cot_id):.4f}")
    if cot_id == "query_id":  # luật đề Nordland: thiếu dòng tính sai, thiếu cột được 0, query_id lặp chỉ chấm dòng đầu
        thieu = mod.score(pub, dung.iloc[: len(dung) // 2], cot_id)
        sai_cot = mod.score(pub, dung.rename(columns={"winter_id": "w"}), cot_id)
        lap = pd.concat([dung.assign(winter_id="w_000000"), dung])
        print(f"   bỏ nửa số dòng: {thieu:.2f}   sai tên cột: {sai_cot:.2f}   dòng đầu sai rồi lặp dòng đúng: {mod.score(pub, lap, cot_id):.2f}")
        continue
    try:
        mod.score(pub, dung.iloc[:-5], cot_id)
        print("   LỖI: thiếu dòng mà không báo")
    except mod.ParticipantVisibleError as e:
        print("   thiếu dòng báo lỗi:", e)
