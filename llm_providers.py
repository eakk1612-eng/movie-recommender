"""llm_providers.py — ระบบสำรอง LLM สำหรับ CineMatch

ลองเรียก AI ทีละเจ้าตามลำดับ ถ้าเจ้าไหนโควตาหมด/ล่ม/คีย์ผิด จะข้ามไปเจ้าถัดไปอัตโนมัติ
เจ้าที่โควตาหมด (429) จะถูกพักไว้ COOLDOWN_SECONDS วินาที ไม่ต้องลองซ้ำทุกครั้งที่ค้นหา
(การพักนับแยกตาม "เจ้า + โมเดล" เพราะโควตาฟรีของแต่ละโมเดลแยกกัน)

เลือกโมเดลเองได้: generate_text(prompt, preferred=(provider_key, model))
  - จะลองโมเดลที่เลือกก่อน ถ้าใช้ไม่ได้ (โควตาหมด/ล่ม/ชื่อโมเดลผิด) สลับไปตามลำดับสำรองปกติให้เอง
  - รายการโมเดลที่เลือกได้ดูจาก model_options()

ตั้งค่าใน .env (ใส่เฉพาะเจ้าที่มีคีย์ — เจ้าที่ไม่มีคีย์จะถูกข้าม):
    GEMINI_API_KEY=...
    GROQ_API_KEY=...          # https://console.groq.com/keys
    OPENROUTER_API_KEY=...    # https://openrouter.ai/keys
    MISTRAL_API_KEY=...       # (ไม่บังคับ) https://console.mistral.ai/api-keys
    LLM_ORDER=gemini,groq,openrouter,mistral,ollama   # (ไม่บังคับ) ลำดับที่จะลอง

AI ในเครื่อง (Ollama) — ไม่ต้องใช้คีย์ ใช้ได้เมื่อรัน `ollama serve` อยู่:
    OLLAMA_MODEL=qwen2.5:7b          # (ไม่บังคับ) ถ้าไม่ตั้ง/ไม่ได้ติดตั้ง จะใช้ตัวแรกที่ติดตั้งไว้
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_NUM_CTX=8192              # ขนาด context (prompt มีรายชื่อหนังยาว ต้องไม่สั้นเกินไป)
    OLLAMA_TIMEOUT=180               # โมเดลในเครื่องช้ากว่าออนไลน์
  - เปิด "โหมดออฟไลน์" (OFFLINE_MODE=1) = ใช้เฉพาะ Ollama ไม่ยิงออกอินเทอร์เน็ตเลย
  - โหมดปกติ Ollama อยู่ท้ายลำดับเป็นตัวสำรอง และถ้าเน็ตหลุด (offline.mark_network_down) จะขึ้นมาก่อน

ชื่อโมเดลเปลี่ยนบ่อย ปรับผ่าน .env ได้โดยไม่ต้องแก้โค้ด:
    GEMINI_MODEL, GROQ_MODEL, OPENROUTER_MODEL, MISTRAL_MODEL, GROQ_STT_MODEL
        -> โมเดลหลักของแต่ละเจ้า (ใช้ในโหมดอัตโนมัติ)
    GEMINI_MODELS, GROQ_MODELS, OPENROUTER_MODELS, MISTRAL_MODELS
        -> (ไม่บังคับ) โมเดลเพิ่มเติมให้เลือกใน dropdown คั่นด้วยคอมมา เช่น
           GEMINI_MODELS=gemini-2.5-pro,gemini-2.5-flash-lite
"""
import os
import time

import requests

import offline  # offline.py ไม่ import ไฟล์นี้กลับ จึงไม่เกิดวงจร import

COOLDOWN_SECONDS = 600
REQUEST_TIMEOUT = 60

_cooldown_until = {}  # "เจ้า:โมเดล" -> เวลา (epoch) ที่จะกลับมาลองใหม่


class AllProvidersFailed(Exception):
    """ทุกผู้ให้บริการ AI ที่ตั้งค่าไว้ใช้ไม่ได้"""


def _env(name, default=""):
    # อ่านตอนเรียกใช้ (ไม่ใช่ตอน import) เพื่อให้ load_dotenv() ในแอปทำงานก่อน
    return os.getenv(name, default)


def _is_quota_error(msg):
    m = msg.lower()
    return "429" in m or "resource_exhausted" in m or "rate limit" in m or "quota" in m


def _cd_key(key, model):
    return f"{key}:{model}"


# ---------------------------------------------------------
# แต่ละเจ้า: รับ (prompt, model) คืนข้อความตอบกลับ
# ---------------------------------------------------------
def _gemini(prompt, model):
    from google import genai

    client = genai.Client(api_key=_env("GEMINI_API_KEY"))
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text or ""


def _openai_compatible(url, api_key, model, prompt, extra_headers=None):
    """Groq / OpenRouter / Mistral ใช้รูปแบบ API แบบเดียวกับ OpenAI"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    res = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4000,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if res.status_code != 200:
        raise RuntimeError(f"{res.status_code} {res.text[:200]}")
    return res.json()["choices"][0]["message"].get("content") or ""


def _groq(prompt, model):
    return _openai_compatible(
        "https://api.groq.com/openai/v1/chat/completions", _env("GROQ_API_KEY"), model, prompt
    )


def _openrouter(prompt, model):
    return _openai_compatible(
        "https://openrouter.ai/api/v1/chat/completions", _env("OPENROUTER_API_KEY"), model, prompt,
        extra_headers={"X-Title": "CineMatch AI"},
    )


def _mistral(prompt, model):
    return _openai_compatible(
        "https://api.mistral.ai/v1/chat/completions", _env("MISTRAL_API_KEY"), model, prompt
    )


# ---------- Ollama (AI ในเครื่อง) ----------
def _ollama_host():
    return _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama(prompt, model):
    try:
        num_ctx = int(_env("OLLAMA_NUM_CTX", "8192"))
    except ValueError:
        num_ctx = 8192
    try:
        timeout = float(_env("OLLAMA_TIMEOUT", "180"))
    except ValueError:
        timeout = 180.0
    res = requests.post(
        f"{_ollama_host()}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # num_ctx สำคัญ: prompt มีรายชื่อหนังยาว ถ้า context สั้นเกินไปจะถูกตัดท้าย
            "options": {"temperature": 0.7, "num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    if res.status_code != 200:
        raise RuntimeError(f"{res.status_code} {res.text[:200]} ({_ollama_host()})")
    return res.json().get("message", {}).get("content") or ""


_local_cache = (0.0, [])


def local_models():
    """ชื่อโมเดลที่ติดตั้งใน Ollama (แคช 5 วินาที เพราะถูกเรียกบ่อยตอนหน้าเว็บ rerun)
    คืนลิสต์ว่างถ้า Ollama ไม่ได้รันอยู่"""
    global _local_cache
    now = time.time()
    if now - _local_cache[0] < 5:
        return _local_cache[1]
    names = []
    try:
        r = requests.get(f"{_ollama_host()}/api/tags", timeout=0.5)
        if r.status_code == 200:
            names = [m["name"] for m in r.json().get("models", []) if m.get("name")]
    except (requests.RequestException, ValueError):
        pass
    _local_cache = (now, names)
    return names


# key -> ข้อมูลของแต่ละเจ้า
#   label          ชื่อที่แสดง
#   key_env        ตัวแปรคีย์ใน .env (เจ้าในเครื่องไม่ใช้)
#   model_env      ตัวแปรชื่อโมเดลหลัก (ใช้ในโหมดอัตโนมัติ)
#   default_model  โมเดลหลักเริ่มต้น
#   extra_models   โมเดลเพิ่มเติมที่ให้เลือกใน dropdown (เพิ่ม/แก้ผ่าน <KEY>_MODELS ใน .env ได้)
#   local          True = ทำงานในเครื่อง ไม่ต้องใช้คีย์ (เช็กว่าพร้อมจากการเรียก local_models())
#   fn             ฟังก์ชันเรียก
_PROVIDERS = {
    "gemini": {
        "label": "Gemini", "key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.5-flash-lite",
        "extra_models": ["gemini-3.8-flash"],
        "fn": _gemini,
    },
    "groq": {
        "label": "Groq", "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
        "extra_models": ["openai/gpt-oss-20b"],
        "fn": _groq,
    },
    "openrouter": {
        "label": "OpenRouter", "key_env": "OPENROUTER_API_KEY", "model_env": "OPENROUTER_MODEL",
        "default_model": "openrouter/free",  # ตัวสุ่มเลือกโมเดลฟรีที่ยังเปิดอยู่ให้เอง ไม่โดน 404 ง่าย
        "extra_models": ["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free"],
        "fn": _openrouter,
    },
    "mistral": {
        "label": "Mistral", "key_env": "MISTRAL_API_KEY", "model_env": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
        "extra_models": ["mistral-large-latest"],
        "fn": _mistral,
    },
    "ollama": {
        "label": "Ollama (ในเครื่อง)", "key_env": "", "model_env": "OLLAMA_MODEL",
        "default_model": "qwen2.5:7b",
        "extra_models": [],
        "local": True,
        "fn": _ollama,
    },
}

_DEFAULT_ORDER = "gemini,groq,openrouter,mistral,ollama"


def _order():
    # โหมดออฟไลน์เต็มรูปแบบ: ใช้เฉพาะ AI ในเครื่อง ไม่ยิงออกอินเทอร์เน็ตเลย
    if offline.forced():
        return ["ollama"]
    raw = _env("LLM_ORDER", _DEFAULT_ORDER).split(",")
    order = list(dict.fromkeys(x.strip().lower() for x in raw if x.strip()))
    # เน็ตหลุดอยู่ (TMDB เรียกไม่ได้): เอา Ollama ขึ้นก่อน จะได้ไม่เสียเวลารอเจ้าออนไลน์ที่ล้มแน่ๆ
    if not offline.network_ok() and "ollama" in order:
        order.remove("ollama")
        order.insert(0, "ollama")
    return order


def _available(key):
    """เจ้านี้พร้อมใช้หรือไม่: เจ้าออนไลน์ = มีคีย์, เจ้าในเครื่อง = Ollama รันอยู่และมีโมเดล"""
    p = _PROVIDERS.get(key)
    if not p:
        return False
    if p.get("local"):
        return bool(local_models())
    return bool(_env(p["key_env"]))


def _primary_model(key):
    p = _PROVIDERS[key]
    if p.get("local"):
        installed = local_models()
        want = _env(p["model_env"], p["default_model"])
        if want in installed:
            return want
        if f"{want}:latest" in installed:
            return f"{want}:latest"
        # โมเดลที่ตั้งไว้ยังไม่ได้ติดตั้ง: ใช้ตัวแรกที่มีอยู่จริงแทน
        return installed[0] if installed else want
    return _env(p["model_env"], p["default_model"])


def _chain():
    """รายการผู้ให้บริการที่พร้อมใช้ เรียงตามลำดับ (เจ้าละ 1 โมเดลหลัก): [(key, label, model, fn)]"""
    out = []
    for key in _order():
        if _available(key):
            p = _PROVIDERS[key]
            out.append((key, p["label"], _primary_model(key), p["fn"]))
    return out


def _models_for(key):
    """โมเดลที่เลือกได้ของเจ้านั้น: โมเดลหลัก + ตัวเสริมในโค้ด + ตัวเสริมจาก <KEY>_MODELS
    (Ollama: โมเดลหลัก + ทุกตัวที่ติดตั้งไว้ในเครื่อง)"""
    p = _PROVIDERS[key]
    if p.get("local"):
        return list(dict.fromkeys([_primary_model(key)] + local_models()))
    extra_env = [m.strip() for m in _env(f"{key.upper()}_MODELS").split(",") if m.strip()]
    return list(dict.fromkeys([_primary_model(key)] + p["extra_models"] + extra_env))


# ---------------------------------------------------------
# ฟังก์ชันที่แอปเรียกใช้
# ---------------------------------------------------------
def has_any_llm():
    return bool(_chain())


def has_groq():
    return bool(_env("GROQ_API_KEY"))


def describe_chain():
    """[(ชื่อเจ้า, ชื่อโมเดล)] ตามลำดับที่จะลองในโหมดอัตโนมัติ — ไว้แสดงบนหน้าเว็บ"""
    return [(label, model) for _key, label, model, _fn in _chain()]


def model_options():
    """[(key, ชื่อเจ้า, โมเดล)] ทุกโมเดลที่ผู้ใช้เลือกได้ (เฉพาะเจ้าที่พร้อมใช้) เรียงตาม LLM_ORDER"""
    out = []
    for key in _order():
        if _available(key):
            label = _PROVIDERS[key]["label"]
            out.extend((key, label, m) for m in _models_for(key))
    return out


def generate_text(prompt, preferred=None):
    """ส่ง prompt ให้ AI คืนค่า (ข้อความ, info)
    info = {"provider": ชื่อเจ้าที่ตอบ, "model": ชื่อโมเดล,
            "skipped": [เจ้า/โมเดลที่ลองหรือข้ามไปแล้วใช้ไม่ได้],
            "chosen": True ถ้าเป็นโมเดลที่ผู้ใช้เลือกเองและตอบได้}

    preferred = (provider_key, model) ที่ผู้ใช้เลือก หรือ None = อัตโนมัติ
    ถ้าตัวที่เลือกใช้ไม่ได้ จะสลับไปตามลำดับสำรองปกติ
    ถ้าล้มเหลวหมดจะ raise AllProvidersFailed"""
    chain = _chain()

    # ผู้ใช้เลือกโมเดลเอง: ลองตัวนั้นก่อน (ลองครั้งเดียว ถ้าโควตาหมดจะล้มเหลวเร็ว)
    # โหมดออฟไลน์: ไม่ยอมให้ตัวที่เลือกเป็นเจ้าออนไลน์ (กันค่าเก่าค้างใน session)
    pref = None
    if preferred:
        pkey, pmodel = preferred
        allowed = (not offline.forced()) or pkey == "ollama"
        if pkey in _PROVIDERS and allowed and _available(pkey) and pmodel:
            pref = (pkey, _PROVIDERS[pkey]["label"], pmodel, _PROVIDERS[pkey]["fn"])

    if not chain and not pref:
        if offline.forced():
            raise AllProvidersFailed(
                "โหมดออฟไลน์: ไม่พบ Ollama ที่รันอยู่ (รัน `ollama serve` แล้ว `ollama pull <ชื่อโมเดล>` "
                f"— ตรวจที่ {_ollama_host()})"
            )
        raise AllProvidersFailed(
            "ยังไม่ได้ตั้งค่า API key ของ AI เจ้าใดเลย และไม่พบ Ollama ในเครื่อง"
        )

    errors, pref_display = [], None
    if pref:
        pkey, plabel, pmodel, pfn = pref
        pref_display = f"{plabel} {pmodel}"
        try:
            text = pfn(prompt, pmodel)
            if text and text.strip():
                _cooldown_until.pop(_cd_key(pkey, pmodel), None)
                return text, {"provider": plabel, "model": pmodel, "skipped": [], "chosen": True}
            errors.append(f"{pref_display}: ตอบกลับว่างเปล่า")
        except Exception as e:
            msg = str(e)
            if _is_quota_error(msg):
                _cooldown_until[_cd_key(pkey, pmodel)] = time.time() + COOLDOWN_SECONDS
            errors.append(f"{pref_display}: {msg[:120]}")
        # ตัวที่เลือกล้มเหลวแล้ว ไม่ต้องลองซ้ำในลำดับสำรอง
        chain = [c for c in chain if not (c[0] == pkey and c[2] == pmodel)]

    now = time.time()
    # เจ้าที่ไม่ได้พักอยู่ก่อน เจ้าที่พักโควตาอยู่เป็นทางเลือกสุดท้าย
    active = [c for c in chain if _cooldown_until.get(_cd_key(c[0], c[2]), 0) <= now]
    resting = [c for c in chain if c not in active]

    for key, label, model, fn in active + resting:
        try:
            text = fn(prompt, model)
            if text and text.strip():
                _cooldown_until.pop(_cd_key(key, model), None)
                # skipped = โมเดลที่ผู้ใช้เลือกแต่ใช้ไม่ได้ + เจ้าที่อยู่ก่อนหน้าเจ้าที่ตอบในลำดับที่ตั้งไว้
                order = [c[1] for c in chain]
                skipped = ([pref_display] if pref_display else []) + order[: order.index(label)]
                return text, {"provider": label, "model": model, "skipped": skipped, "chosen": False}
            errors.append(f"{label}: ตอบกลับว่างเปล่า")
        except Exception as e:  # โควตาหมด, 404, ล่ม, คีย์ผิด ฯลฯ -> ลองเจ้าถัดไป
            msg = str(e)
            if _is_quota_error(msg):
                _cooldown_until[_cd_key(key, model)] = time.time() + COOLDOWN_SECONDS
            errors.append(f"{label}: {msg[:120]}")
    raise AllProvidersFailed("ใช้ AI ไม่ได้ทุกเจ้า — " + " | ".join(errors))


def groq_transcribe(audio_bytes, mime_type="audio/wav", hint=""):
    """ถอดเสียงด้วย Whisper บน Groq (สำรองเมื่อ Gemini ถอดเสียงไม่ได้)"""
    ext = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/webm": "webm",
           "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/ogg": "ogg"}.get(mime_type, "wav")
    data = {"model": _env("GROQ_STT_MODEL", "whisper-large-v3-turbo"), "response_format": "text"}
    if hint:
        data["prompt"] = hint[:400]
    res = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {_env('GROQ_API_KEY')}"},
        files={"file": (f"audio.{ext}", audio_bytes, mime_type)},
        data=data,
        timeout=REQUEST_TIMEOUT,
    )
    if res.status_code != 200:
        raise RuntimeError(f"{res.status_code} {res.text[:200]}")
    return res.text.strip()