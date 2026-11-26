"""Kiểm tra metric của từng bài sau khi dựng đề:
- nộp chính đáp án (bỏ Usage, đổi tên cột cho đúng định dạng nộp) phải ra điểm tối đa;
- nộp file mẫu phải chấm được;
- nộp thiếu dòng phải báo ParticipantVisibleError (thí sinh thấy được lỗi).
Ghi kèm <Bài>/host/nop_dung_het.csv để nộp thử trên Kaggle (Sandbox Submission phải ra điểm tối đa).
    python tools/kiem_tra_metric.py Nordland_PlaceRecognition VSFC_Sentiment ...
"""
import importlib.util
import json
import os
import sys

import numpy as np

import pandas as pd

from chung import GOC

DOI_TEN = {"positives": "predictions", "passage_id": "passage_ids"}
for b in sys.argv[1:]:
    host = GOC / b / "host"
    spec = importlib.util.spec_from_file_location("metric_bai", host / "metric.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sol = pd.read_csv(host / "solution.csv", dtype=str, keep_default_na=False)
    dung = sol.drop(columns=[c for c in ("Usage", "height", "width") if c in sol]).rename(columns=DOI_TEN)
    if "label" in dung and (b.startswith("MVTec") or b.startswith("HomeCredit")):
        dung = dung.rename(columns={"label": "score"})  # AUROC: nộp cột score
    if b.startswith("ViQuAD_Extractive"):
        dung = pd.DataFrame({"id": sol.id, "answer": [json.loads(a)[0] for a in sol.answers]})
    if b.startswith("KTVIC"):  # nộp câu tham chiếu đầu tiên: điểm cao nhưng không tối đa (CIDEr so với cả 5 câu)
        dung = pd.DataFrame({"id": sol.id, "caption": [json.loads(a)[0] for a in sol.captions]})
    if b.startswith("Robot"):  # lời giải tối ưu tính lại bằng Dijkstra + quy hoạch động
        os.environ.setdefault("SELFHOST_SEED", "0")
        from xay_de_them import giai_toi_uu
        dung = pd.DataFrame({"id": sol.id, "actions": [giai_toi_uu(g.split("/"))[1] for g in sol.grid]})
    if b.startswith("CIFAR_Adv"):  # không có "đáp án đúng hết"; nộp ảnh gốc phải được 0, lệch 9 mức xám phải không hợp lệ
        print(f"{b:28s} ảnh gốc: {mod.score(sol, sol.rename(columns={'orig': 'image'})[['id', 'image']], 'id'):.4f}", end="   ")
        import base64
        lech = [base64.b64encode(np.clip(np.frombuffer(base64.b64decode(s), np.uint8).astype(int) + 9, 0, 255).astype(np.uint8).tobytes()).decode() for s in sol.orig]
        print(f"lệch 9 mức (không hợp lệ): {mod.score(sol, pd.DataFrame({'id': sol.id, 'image': lech}), 'id'):.4f}")
        continue
    cot_id = "id"
    if b.startswith("Nordland"):  # nộp một winter_id đúng cho mỗi ảnh mùa hè
        dung = pd.DataFrame({"id": sol.id, "winter_id": sol.winter_id.str.split().str[0]})
    dung.to_csv(host / "nop_dung_het.csv", index=False)
    mau = pd.read_csv(GOC / b / "upload/sample_submission.csv", dtype=str, keep_default_na=False)
    pub = sol[sol.Usage == "Public"].reset_index(drop=True)
    print(f"{b:28s} đúng hết: {mod.score(pub, dung, cot_id):.4f}   file mẫu: {mod.score(pub, mau, cot_id):.4f}")
    if b.startswith("Nordland"):  # luật đề Nordland: thiếu dòng tính sai, thiếu cột được 0, id lặp chỉ chấm dòng đầu
        thieu = mod.score(pub, dung.iloc[: len(dung) // 2], cot_id)
        sai_cot = mod.score(pub, dung.rename(columns={"winter_id": "w"}), cot_id)
        lap = pd.concat([dung.assign(winter_id="w_000000"), dung])
        print(f"   bỏ nửa số dòng: {thieu:.2f}   sai tên cột: {sai_cot:.2f}   dòng đầu sai rồi lặp dòng đúng: {mod.score(pub, lap, cot_id):.2f}")
        continue
    if b.startswith("DimABSA"):  # thiếu câu không báo lỗi: thiếu khía cạnh bị tính sai số lớn nhất, thiếu bộ tứ tính bỏ sót
        nua = dung[dung[cot_id].isin(pub[cot_id])].iloc[::2]
        print(f"   bỏ nửa số câu public: {mod.score(pub, nua, cot_id):.4f}   sai tên cột: ", end="")
        try:
            mod.score(pub, dung.rename(columns={dung.columns[1]: "x"}), cot_id); print("LỖI: không báo")
        except mod.ParticipantVisibleError as e:
            print("báo lỗi:", e)
        continue
    try:
        mod.score(pub, dung.iloc[:-5], cot_id)
        print("   LỖI: thiếu dòng mà không báo")
    except mod.ParticipantVisibleError as e:
        print("   thiếu dòng báo lỗi:", e)
