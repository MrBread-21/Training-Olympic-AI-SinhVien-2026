#!/bin/bash
# Chạy baseline.py của từng bài trên upload/, chấm, ghi điểm vào <Bài>/host/min_score.txt (Min_Score khi quy đổi).
#   bash tools/chay_baseline.sh VSFC_Sentiment PhoNER_COVID19 ...
export PYTHONIOENCODING=utf-8
GOC="$(cd "$(dirname "$0")/.." && pwd)"
for b in "$@"; do
  t0=$(date +%s)
  (cd "$GOC/$b" && python baseline.py upload > host/baseline.log 2>&1) || { echo "$b LỖI"; tail -5 "$GOC/$b/host/baseline.log"; continue; }
  nop="$GOC/$b/submission.csv"; [ -f "$GOC/$b/output.csv" ] && nop="$GOC/$b/output.csv"  # Nordland ghi output.csv như đề
  python "$GOC/tools/cham.py" "$b" "$nop" > "$GOC/$b/host/min_score.txt"
  echo "== $b ($(( $(date +%s) - t0 ))s)"; cat "$GOC/$b/host/min_score.txt"
done
