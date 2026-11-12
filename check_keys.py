"""check_keys.py — ทดสอบว่าแต่ละเจ้า/โมเดลใช้ได้จริงไหม และบอกสาเหตุที่ล้มเหลว

วิธีใช้ (รันในโฟลเดอร์เดียวกับ app.py):
    python check_keys.py

ไม่พิมพ์คีย์ออกหน้าจอ แสดงเฉพาะผลและข้อความ error จากผู้ให้บริการ
"""
import time

from dotenv import load_dotenv

load_dotenv()

import llm_providers as L  # noqa: E402  (ต้องโหลด .env ก่อน)

options = L.model_options()
if not options:
    print("ไม่พบ API key ใน .env เลย")
    raise SystemExit(1)

for key, label, model in options:
    fn = L._PROVIDERS[key]["fn"]
    start = time.time()
    try:
        text = fn("ตอบสั้นๆ ว่า OK", model)
        secs = time.time() - start
        if text and text.strip():
            print(f"✅ {label} · {model}  ({secs:.1f} วินาที)")
        else:
            print(f"⚠️  {label} · {model}  ตอบกลับว่างเปล่า")
    except Exception as e:
        print(f"❌ {label} · {model}\n     สาเหตุ: {str(e)[:300]}")