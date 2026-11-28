"""Gửi prompt tới chatbot mô phỏng phòng thi, ghép code trả về thành một file chạy được.

    set CHATBOT_URL=https://<địa-chỉ>/chatbot/        (PowerShell: $env:CHATBOT_URL = "...")
    python hoi_llm.py <file_ra.py> <prompt_1.txt> [prompt_2.txt ...] [--dau]

- Mỗi prompt được gửi trong MỘT cuộc trò chuyện mới (không kèm lịch sử), giống bấm "Cuộc trò chuyện mới".
- Khối ```python dài nhất của mỗi câu trả lời được nối theo thứ tự prompt; --dau thêm dau_file.py lên đầu.
- Câu trả lời đầy đủ lưu cạnh file ra (<file_ra>.p1.txt, .p2.txt, ...) để đọc lại.
- In số token đã dùng / 2.000 và cảnh báo khi câu trả lời bị cắt vì hết token.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) < 2:
    print(__doc__)
    sys.exit(1)
url = os.environ.get("CHATBOT_URL", "").rstrip("/")
if not url:
    print("Chưa đặt biến môi trường CHATBOT_URL (địa chỉ trang chatbot, ví dụ https://.../chatbot)")
    sys.exit(1)
API = url + "/api/chat"
HDR = {"User-Agent": "Mozilla/5.0", "ngrok-skip-browser-warning": "1",
       "Content-Type": "application/json", "Accept": "application/x-ndjson"}
out = Path(args[0])
phan = []
for k, p in enumerate(args[1:], 1):
    prompt = Path(p).read_text(encoding="utf-8").strip()
    body = json.dumps({"content": prompt, "history": []}, ensure_ascii=False).encode("utf-8")
    t0, ans, done = time.time(), [], {}
    with urllib.request.urlopen(urllib.request.Request(API, data=body, headers=HDR, method="POST"), timeout=900) as r:
        for line in r:
            if line.strip():
                ev = json.loads(line)
                if "delta" in ev:
                    ans.append(ev["delta"])
                elif "done" in ev:
                    done = ev["done"]
                elif "error" in ev:
                    print(f"Lỗi từ chatbot: {ev['error']}")
                    sys.exit(1)
    text = done.get("answer") or "".join(ans)
    out.with_suffix(f".p{k}.txt").write_text(text, encoding="utf-8")
    khoi = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    code = max(khoi, key=len) if khoi else re.sub(r"^```(?:python|py)?[ \t]*\n", "", text.strip())
    phan.append(code.rstrip() + "\n")
    canh_bao = "  BỊ CẮT vì hết token: rút gọn prompt hoặc chia nhỏ hơn" if done.get("limited") else ""
    print(f"{Path(p).name}: {time.time() - t0:.0f} giây, {done.get('used')} / 2.000 token, "
          f"prompt {len(prompt)} ký tự, code {len(code.splitlines())} dòng{canh_bao}")
dau = (Path(__file__).with_name("dau_file.py").read_text(encoding="utf-8") + "\n\n") if "--dau" in sys.argv else ""
out.write_text(dau + "\n\n".join(phan), encoding="utf-8")
print(f"ghi {out}")
