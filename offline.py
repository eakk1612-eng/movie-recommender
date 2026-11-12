"""offline.py — ระบบออฟไลน์ของ CineMatch

ทำให้แอปทำงานได้โดยไม่ใช้อินเทอร์เน็ต ประกอบด้วย
  1) ข้อมูลหนังในเครื่อง  : อ่านจาก data/catalog.json + รูปใน static/ (สร้างด้วย build_offline_data.py)
                           ตอบคำถามแบบเดียวกับ TMDB API (fetch / details) แอปจึงแทบไม่ต้องแก้โค้ดส่วนอื่น
  2) ค้นชื่อหนังในเครื่อง  : ค้นแบบทนสะกดผิด/ถอดเสียงผิด
  3) รายชื่อหนังให้ AI เลือก: AI เล็กๆ ในเครื่องจำชื่อหนังได้ไม่แม่น จึงให้เลือกจากรายชื่อที่คัดไว้
  4) ถอดเสียงในเครื่อง     : faster-whisper (ต้องดาวน์โหลดโมเดลไว้ก่อนด้วย build_offline_data.py --whisper)

โหมดการทำงาน (ตั้งใน .env ด้วย OFFLINE_MODE=1 หรือสลับในหน้า "ตั้งค่า AI")
  - บังคับออฟไลน์ (forced): ไม่เรียก TMDB / Gemini / Groq เลย ใช้ของในเครื่องล้วน
  - อัตโนมัติ: ปกติใช้ออนไลน์ ถ้าไม่มีคีย์ TMDB หรืออินเทอร์เน็ตหลุด จะสลับมาใช้ข้อมูลในเครื่องเอง
"""
import importlib.util
import io
import json
import math
import os
import random
import re
import threading
import time
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
STATIC_DIR = BASE_DIR / "static"
POSTER_DIR = STATIC_DIR / "posters"
BACKDROP_DIR = STATIC_DIR / "backdrops"
MODELS_DIR = BASE_DIR / "models"
STATIC_URL = "app/static"  # Streamlit เสิร์ฟไฟล์ในโฟลเดอร์ static/ ที่ /app/static/ (ต้องเปิด enableStaticServing)
NETWORK_RETRY_SECONDS = 60  # หลังเรียก TMDB ไม่ได้ จะพักการเรียกไว้กี่วินาที

try:  # ให้ Streamlit เห็นโฟลเดอร์ static ตั้งแต่เริ่มรัน
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    BACKDROP_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

GENRE_TH = {
    28: "แอคชั่น", 12: "ผจญภัย", 16: "แอนิเมชัน", 35: "ตลก", 80: "อาชญากรรม",
    99: "สารคดี", 18: "ดราม่า", 10751: "ครอบครัว", 14: "แฟนตาซี", 36: "ประวัติศาสตร์",
    27: "สยองขวัญ", 10402: "ดนตรี", 9648: "ลึกลับ", 10749: "โรแมนติก",
    878: "ไซไฟ", 53: "ระทึกขวัญ", 10752: "สงคราม", 37: "คาวบอย",
}

# คำในคำค้น -> แนวหนัง / ภาษา (ใช้คัดรายชื่อหนังให้ AI เมื่อทำงานออฟไลน์)
GENRE_KEYWORDS = {
    35: ["ตลก", "ฮา", "ขำ", "comedy", "funny"],
    27: ["ผี", "สยอง", "หลอน", "ขนลุก", "horror", "scary"],
    878: ["ไซไฟ", "อวกาศ", "อนาคต", "หุ่นยนต์", "sci-fi", "scifi", "sci fi", "space"],
    10749: ["รัก", "โรแมนติก", "romance", "romantic"],
    18: ["ดราม่า", "ซึ้ง", "เรียกน้ำตา", "ร้องไห้", "drama"],
    28: ["แอคชั่น", "แอ็คชั่น", "ต่อสู้", "มันส์", "action"],
    16: ["แอนิเมชัน", "แอนิเมชั่น", "การ์ตูน", "อนิเมะ", "animation", "anime"],
    9648: ["สืบสวน", "ปริศนา", "ลึกลับ", "mystery", "detective"],
    80: ["อาชญากรรม", "มาเฟีย", "แก๊ง", "ฆาตกร", "crime"],
    53: ["ระทึก", "ตื่นเต้น", "หักมุม", "thriller", "suspense"],
    10751: ["ครอบครัว", "เด็ก", "family", "kids"],
    10752: ["สงคราม", "war"],
    12: ["ผจญภัย", "adventure"],
    14: ["แฟนตาซี", "เวทมนตร์", "fantasy", "magic"],
    36: ["ประวัติศาสตร์", "เรื่องจริง", "history", "biography"],
    10402: ["เพลง", "ดนตรี", "มิวสิคัล", "music", "musical"],
    99: ["สารคดี", "documentary"],
    37: ["คาวบอย", "western"],
}
LANG_KEYWORDS = {
    "th": ["หนังไทย", "ไทย", "thai"],
    "ko": ["เกาหลี", "korean", "korea"],
    "ja": ["ญี่ปุ่น", "japan", "japanese"],
    "zh": ["จีน", "ฮ่องกง", "chinese", "hong kong"],
    "en": ["ฮอลลีวูด", "อเมริกา", "อเมริกัน", "hollywood", "american"],
}

# ---------------------------------------------------------
# โหมด / สถานะเครือข่าย
# ---------------------------------------------------------
_forced = None  # อ่านจาก .env ตอนเรียกใช้ครั้งแรก (ไม่อ่านตอน import เพราะ load_dotenv() ในแอปทำงานทีหลัง)
_down_until = 0.0


def forced():
    """True = โหมดออฟไลน์เต็มรูปแบบ (ไม่แตะอินเทอร์เน็ตเลย)"""
    global _forced
    if _forced is None:
        _forced = os.getenv("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes", "on")
    return _forced


def set_forced(value):
    global _forced
    _forced = bool(value)


def mark_network_down():
    """เรียก TMDB ไม่ได้ (เน็ตหลุด) พักการเรียกชั่วคราว แล้วใช้ข้อมูลในเครื่องแทน (ถ้ามี)"""
    global _down_until
    _down_until = time.time() + NETWORK_RETRY_SECONDS


def network_ok():
    return time.time() >= _down_until


def use_local(tmdb_key=None):
    """ควรใช้ข้อมูลในเครื่องแทน TMDB หรือไม่"""
    if forced():
        return True
    if not catalog_available():
        return False
    return (not tmdb_key) or (not network_ok())


# ---------------------------------------------------------
# โหลดข้อมูลหนังในเครื่อง
# ---------------------------------------------------------
def _default_key(text):
    return re.sub(r"[\s\-_:.,!?'\"“”()]+", "", (text or "").lower())


_keyfunc = _default_key
_lock = threading.RLock()
_cat = {"mtime": None, "movies": [], "meta": {}, "by_id": {}, "keys": {}}


def set_keyfunc(fn):
    """ตั้งฟังก์ชันแปลงชื่อเป็น 'รหัสเสียง' (แอปส่ง phonetic_key มาให้ เพื่อค้นชื่อไทยแบบทนสะกดผิด)"""
    global _keyfunc
    with _lock:
        _keyfunc = fn or _default_key
        _cat["keys"] = {}


def _load():
    try:
        mtime = CATALOG_PATH.stat().st_mtime
    except OSError:
        with _lock:
            _cat.update(mtime=None, movies=[], meta={}, by_id={}, keys={})
        return
    with _lock:
        if _cat["mtime"] == mtime:
            return
        try:
            data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            movies = [m for m in data.get("movies", []) if isinstance(m, dict) and m.get("id")]
        except Exception:
            return
        _cat.update(
            mtime=mtime, movies=movies, meta=data.get("meta", {}),
            by_id={m["id"]: m for m in movies}, keys={},
        )


def catalog_available():
    _load()
    return bool(_cat["movies"])


def catalog_movies():
    _load()
    return _cat["movies"]


def catalog_info():
    _load()
    posters = len(list(POSTER_DIR.glob("*.jpg"))) if POSTER_DIR.exists() else 0
    return {
        "count": len(_cat["movies"]),
        "built_at": str(_cat["meta"].get("built_at", "")),
        "posters": posters,
    }


def _keys(m):
    k = _cat["keys"].get(m["id"])
    if k is None:
        k = [_keyfunc(n) for n in (m.get("title"), m.get("original_title"), m.get("title_en")) if n]
        _cat["keys"][m["id"]] = k
    return k


def _year_ok(m, year, tol):
    try:
        y = int(str(m.get("year") or str(m.get("release_date") or "")[:4]))
        return abs(y - int(year)) <= tol
    except (TypeError, ValueError):
        return True  # ไม่รู้ปี ไม่ตัดทิ้ง


def _result(m):
    """รูปแบบเดียวกับผลลัพธ์ของ TMDB (แอปอ่านเฉพาะบางฟิลด์)"""
    return {
        "id": m["id"],
        "title": m.get("title") or m.get("original_title") or "",
        "original_title": m.get("original_title") or "",
        "release_date": m.get("release_date") or "",
        "popularity": m.get("popularity") or 0,
        "vote_average": m.get("vote_average") or 0,
        "vote_count": m.get("vote_count") or 0,
        "genre_ids": list(m.get("genre_ids", [])),
        "original_language": m.get("original_language") or "",
        "overview": m.get("overview") or "",
        "poster_path": None,
    }


# ---------------------------------------------------------
# เลียนแบบ TMDB API: discover / trending / top_rated / now_playing / search
# ---------------------------------------------------------
def _sorted(rows, sort_by):
    key, _, direction = (sort_by or "popularity.desc").partition(".")
    reverse = direction != "asc"
    if key == "primary_release_date":
        return sorted(rows, key=lambda m: m.get("release_date") or "", reverse=reverse)
    if key == "vote_average":
        return sorted(rows, key=lambda m: m.get("vote_average") or 0, reverse=reverse)
    if key == "vote_count":
        return sorted(rows, key=lambda m: m.get("vote_count") or 0, reverse=reverse)
    return sorted(rows, key=lambda m: m.get("popularity") or 0, reverse=reverse)


def _discover(movies, p):
    lang = p.get("with_original_language")
    gp = str(p.get("with_genres") or "")
    or_set = {int(g) for g in gp.split("|") if g.strip().isdigit()} if "|" in gp else set()
    and_set = {int(g) for g in gp.split(",") if g.strip().isdigit()} if "|" not in gp else set()
    try:
        min_votes = int(float(p.get("vote_count.gte") or 0))
    except ValueError:
        min_votes = 0
    gte, lte = p.get("primary_release_date.gte"), p.get("primary_release_date.lte")

    rows = []
    for m in movies:
        if lang and m.get("original_language") != lang:
            continue
        gids = set(m.get("genre_ids", []))
        if or_set and not (gids & or_set):
            continue
        if and_set and not and_set <= gids:
            continue
        if (m.get("vote_count") or 0) < min_votes:
            continue
        rd = m.get("release_date") or ""
        if gte and rd < gte:
            continue
        if lte and rd > lte:
            continue
        rows.append(m)
    return _sorted(rows, p.get("sort_by"))


def fetch(endpoint, params=None):
    """คืนรายการหนังรูปแบบเดียวกับ TMDB (20 เรื่องต่อหน้า) รองรับ endpoint ที่แอปใช้"""
    p = params or {}
    movies = catalog_movies()
    if not movies:
        return []
    try:
        page = max(1, int(p.get("page", 1) or 1))
    except (TypeError, ValueError):
        page = 1
    today = date.today().isoformat()

    if endpoint == "search/movie":
        return [_result(m) for m in search_loose(p.get("query", ""), p.get("year"))][:20]
    if endpoint == "trending/movie/week":
        rows = _sorted(movies, "popularity.desc")
    elif endpoint == "movie/top_rated":
        rows = [m for m in movies if (m.get("vote_count") or 0) >= 300]
        if len(rows) < 20:
            rows = [m for m in movies if (m.get("vote_count") or 0) >= 50]
        rows = _sorted(rows, "vote_average.desc")
    elif endpoint == "movie/now_playing":
        since = (date.today() - timedelta(days=75)).isoformat()
        rows = _sorted([m for m in movies if since <= (m.get("release_date") or "") <= today], "popularity.desc")
        if not rows:  # ข้อมูลเก่าแล้ว: เอาเรื่องที่ออกล่าสุดในชุดข้อมูลแทน
            rows = _sorted([m for m in movies if (m.get("release_date") or "") <= today], "primary_release_date.desc")
    elif endpoint == "discover/movie":
        rows = _discover(movies, p)
    else:
        return []
    start = (page - 1) * 20
    return [_result(m) for m in rows[start:start + 20]]


def details(movie_id):
    """รายละเอียดหนัง รูปแบบเดียวกับที่ get_movie_details ของแอปคืนค่า (รูปชี้ไปที่ไฟล์ในเครื่อง)"""
    _load()
    try:
        m = _cat["by_id"].get(int(movie_id))
    except (TypeError, ValueError):
        return None
    if not m:
        return None
    poster = f"{STATIC_URL}/posters/{m['id']}.jpg" if (POSTER_DIR / f"{m['id']}.jpg").exists() else None
    backdrop = f"{STATIC_URL}/backdrops/{m['id']}.jpg" if (BACKDROP_DIR / f"{m['id']}.jpg").exists() else None
    return {
        "id": m["id"],
        "title": m.get("title") or m.get("original_title") or "ไม่ทราบชื่อเรื่อง",
        "overview": m.get("overview") or "ไม่มีข้อมูลเรื่องย่อภาษาไทย",
        "rating": round(m.get("vote_average") or 0, 1),
        "release_date": str(m.get("year") or str(m.get("release_date") or "")[:4] or "N/A"),
        "runtime": m.get("runtime") or 0,
        "genres": list(m.get("genres", []))[:3],
        "poster": poster,
        "backdrop": backdrop,
        "trailer": None,  # ตัวอย่างหนังอยู่บน YouTube ใช้ออฟไลน์ไม่ได้
        "providers": list(m.get("providers", [])),
        "rent_buy": list(m.get("rent_buy", [])),
        "language": m.get("original_language"),
    }


# ---------------------------------------------------------
# ค้นชื่อหนังในเครื่อง
# ---------------------------------------------------------
def _best_score(qk, keys, strict):
    """ความคล้าย 0-1 ระหว่างคำค้น (รหัสเสียง) กับชื่อทั้งหมดของหนัง
    strict=False : คำค้นเป็นส่วนหนึ่งของชื่อก็ถือว่าตรง (ค้นภาคต่อ/แฟรนไชส์)
    strict=True  : ใช้จับคู่ชื่อที่ AI เสนอ ต้องเกือบตรงทั้งชื่อเท่านั้น กันจับผิดเรื่อง"""
    best = 0.0
    for nk in keys:
        if not nk:
            continue
        if qk == nk:
            return 1.0
        if not strict and len(qk) >= 5 and qk in nk:
            best = max(best, 0.9)
            continue
        sm = SequenceMatcher(None, qk, nk, autojunk=False)
        if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
            continue
        best = max(best, sm.ratio())
    return best


def search_loose(query, year=None, limit=20, min_score=0.72):
    qk = _keyfunc(query or "")
    if len(qk) < 2:
        return []
    scored = []
    for m in catalog_movies():
        if year and not _year_ok(m, year, 0):
            continue
        sc = _best_score(qk, _keys(m), strict=False)
        if sc >= min_score:
            scored.append((sc, m.get("popularity") or 0, m))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [m for _s, _p, m in scored[:limit]]


def resolve_title(title, year=None, thai_title=None, min_score=0.85):
    """หา id หนังจากชื่อที่ AI เสนอ (เข้มงวด: ไม่เจอก็คืน None ดีกว่าจับผิดเรื่อง)"""
    for name, y in ((title, year), (title, None), (thai_title, None)):
        if not name:
            continue
        qk = _keyfunc(name)
        if len(qk) < 2:
            continue
        best = None
        for m in catalog_movies():
            if y and not _year_ok(m, y, 1):
                continue
            sc = _best_score(qk, _keys(m), strict=True)
            if sc >= min_score:
                rank = (sc, m.get("popularity") or 0)
                if best is None or rank > best[0]:
                    best = (rank, m["id"])
        if best:
            return best[1]
    return None


# ---------------------------------------------------------
# คัดรายชื่อหนังให้ AI เลือก (โหมดออฟไลน์)
# ---------------------------------------------------------
def candidates(query, n=60, thai_min=0):
    """คัดหนังที่น่าจะเข้ากับคำค้นจากข้อมูลในเครื่อง ไว้ให้ AI เลือก
    คืนค่า [{id, title, year, genres}] — ถ้าคำค้นไม่มีสัญญาณอะไร (เช่น "สุ่มให้เลย") จะสุ่มจากหนังเด่นๆ"""
    movies = catalog_movies()
    if not movies:
        return []
    q = (query or "").lower()
    genres = {gid for gid, kws in GENRE_KEYWORDS.items() if any(k in q for k in kws)}
    langs = {code for code, kws in LANG_KEYWORDS.items() if any(k in q for k in kws)}
    words = [w for w in re.split(r"[\s,.!?;:()\"'“”]+", q) if len(w) >= 3]

    scored = []
    for m in movies:
        sig = 3.0 * len(genres & set(m.get("genre_ids", [])))
        if langs and m.get("original_language") in langs:
            sig += 4.0
        if words:
            hay = " ".join(
                str(m.get(k) or "") for k in ("title", "original_title", "title_en", "overview")
            ).lower()
            sig += sum(1.0 for w in words if w in hay)
        prior = 0.4 * math.log10(1 + (m.get("popularity") or 0)) + 0.15 * (m.get("vote_average") or 0)
        scored.append((sig, prior, m))

    if max(s for s, _p, _m in scored) <= 0:  # ไม่มีสัญญาณเลย: สุ่มจากหนังเด่น ให้ได้ผลหลากหลาย
        top = [m for _s, _p, m in sorted(scored, key=lambda t: t[1], reverse=True)[:200]]
        picked = random.sample(top, min(n, len(top)))
    else:
        ranked = [m for _s, _p, m in sorted(scored, key=lambda t: (t[0] + t[1]), reverse=True)]
        picked = ranked[:n]
        # ให้มีหนังไทยอย่างน้อย thai_min เรื่อง เว้นแต่ผู้ใช้ระบุชาติอื่นไว้แล้ว
        if thai_min and not (langs and "th" not in langs):
            have = sum(1 for m in picked if m.get("original_language") == "th")
            if have < thai_min:
                ids = {m["id"] for m in picked}
                extra = [m for m in ranked if m.get("original_language") == "th" and m["id"] not in ids]
                extra = extra[: thai_min - have]
                picked = picked[: max(0, n - len(extra))] + extra

    out = []
    for m in picked:
        title = m.get("title") or m.get("original_title") or ""
        alt = m.get("title_en") or m.get("original_title") or ""
        if alt and alt.lower() != title.lower():
            title = f"{title} / {alt}"
        out.append({
            "id": m["id"],
            "title": title,
            "year": str(m.get("year") or str(m.get("release_date") or "")[:4]),
            "genres": ", ".join(GENRE_TH[g] for g in m.get("genre_ids", []) if g in GENRE_TH),
        })
    return out


# ---------------------------------------------------------
# ถอดเสียงในเครื่อง (faster-whisper)
# ---------------------------------------------------------
_whisper = None
_wlock = threading.Lock()


def whisper_name():
    return os.getenv("WHISPER_MODEL", "small")


def whisper_installed():
    return importlib.util.find_spec("faster_whisper") is not None


def whisper_ready():
    """ติดตั้ง faster-whisper แล้ว และมีโมเดลดาวน์โหลดไว้ในเครื่องแล้วหรือยัง"""
    if not whisper_installed():
        return False
    name = whisper_name()
    if Path(name).exists():
        return True
    return MODELS_DIR.exists() and any(MODELS_DIR.glob(f"models--*{name}*"))


def _get_whisper():
    global _whisper
    with _wlock:
        if _whisper is None:
            if not whisper_installed():
                raise RuntimeError("ยังไม่ได้ติดตั้ง faster-whisper (pip install faster-whisper)")
            from faster_whisper import WhisperModel

            try:
                _whisper = WhisperModel(
                    whisper_name(),
                    device=os.getenv("WHISPER_DEVICE", "auto"),
                    compute_type=os.getenv("WHISPER_COMPUTE", "int8"),
                    download_root=str(MODELS_DIR),
                    local_files_only=True,  # ออฟไลน์: ห้ามพยายามดาวน์โหลด
                )
            except Exception as e:
                raise RuntimeError(
                    "ยังไม่มีโมเดลถอดเสียงในเครื่อง ให้รัน python build_offline_data.py --whisper "
                    f"{whisper_name()} ตอนมีอินเทอร์เน็ตก่อน ({str(e)[:100]})"
                ) from e
        return _whisper


def local_transcribe(audio_bytes, hint=""):
    """ถอดเสียงด้วย Whisper ในเครื่อง คืนข้อความ (ว่างถ้าไม่ได้ยินเสียงพูด)"""
    model = _get_whisper()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=(os.getenv("WHISPER_LANGUAGE", "th") or None),
        initial_prompt=(hint[:400] or None),
        vad_filter=True,
        beam_size=5,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def download_whisper(name=None):
    """ดาวน์โหลดโมเดลถอดเสียงมาเก็บในเครื่อง (ต้องมีอินเทอร์เน็ต ทำครั้งเดียว)"""
    if not whisper_installed():
        raise RuntimeError("ยังไม่ได้ติดตั้ง faster-whisper (pip install faster-whisper)")
    from faster_whisper import WhisperModel

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WhisperModel(name or whisper_name(), device="cpu", compute_type="int8", download_root=str(MODELS_DIR))