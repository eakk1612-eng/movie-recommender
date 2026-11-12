#!/usr/bin/env python3
"""build_offline_data.py — เตรียมข้อมูลสำหรับใช้งานแบบออฟไลน์ (รันตอนมีอินเทอร์เน็ต ทำครั้งเดียว อัปเดตเมื่อไรก็ได้)

    python build_offline_data.py                  # ดาวน์โหลดข้อมูลหนัง ~2000 เรื่อง + โปสเตอร์ (ต้องมี TMDB_API_KEY)
    python build_offline_data.py --size 1000      # เลือกจำนวนหนัง
    python build_offline_data.py --whisper small  # ดาวน์โหลดโมเดลถอดเสียงไว้ในเครื่องด้วย (ต้อง pip install faster-whisper)
    python build_offline_data.py --skip-catalog --whisper small   # โหลดเฉพาะโมเดลถอดเสียง
    python build_offline_data.py --status         # ตรวจว่าส่วนออฟไลน์พร้อมใช้แค่ไหน (ไม่ต้องใช้อินเทอร์เน็ต)

สิ่งที่ได้
    data/catalog.json        ข้อมูลหนัง (ชื่อ เรื่องย่อ คะแนน แนว สตรีมมิ่งในไทย ฯลฯ)
    static/posters/*.jpg     โปสเตอร์ (w342)
    static/backdrops/*.jpg   ภาพฉากหลังของหนังเด่น 150 เรื่องแรก (ใช้ใน Hero)
    models/                  โมเดลถอดเสียง Whisper (ถ้าใช้ --whisper)

รันซ้ำได้: เรื่องที่มีอยู่แล้วและรูปที่ดาวน์โหลดแล้วจะถูกข้าม (ใช้ --refresh เพื่อดึงข้อมูลใหม่ทั้งหมด)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv

import offline

load_dotenv()
TMDB = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p"
KEY = os.getenv("TMDB_API_KEY")
WORKERS = 8
BACKDROP_TOP = 150  # จำนวนหนังเด่นที่โหลดภาพฉากหลังไว้ใช้ใน Hero

GENRES = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 53, 10752, 37]
# น้ำหนักของแต่ละแหล่งตอนผสมรายชื่อ (หนังไทยได้สัดส่วนมากกว่า)
WEIGHTS = {"thai": 3, "thai_horror": 1, "thai_recent": 1, "popular": 2, "top_rated": 2}


def tmdb_get(path, **params):
    for attempt in range(4):
        try:
            r = requests.get(f"{TMDB}/{path}", params={"api_key": KEY, **params}, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 401:
                sys.exit("❌ TMDB_API_KEY ไม่ถูกต้อง (401)")
            return None
        except requests.RequestException:
            time.sleep(1 + attempt)
    return None


# ---------- 1) รวบรวมรายชื่อหนัง ----------
def source_jobs():
    today = date.today()
    recent = (today - timedelta(days=120)).isoformat()
    four_years = (today - timedelta(days=365 * 4)).isoformat()
    th = {"with_original_language": "th", "sort_by": "popularity.desc"}
    src = {
        "popular": [("discover/movie", {"sort_by": "popularity.desc", "vote_count.gte": 500, "page": p}) for p in range(1, 31)],
        "top_rated": [("movie/top_rated", {"page": p}) for p in range(1, 26)],
        "thai": [("discover/movie", {**th, "vote_count.gte": 3, "page": p}) for p in range(1, 26)],
        "thai_horror": [("discover/movie", {**th, "with_genres": "27", "vote_count.gte": 3, "page": p}) for p in range(1, 7)],
        "thai_recent": [("discover/movie", {
            "with_original_language": "th", "sort_by": "primary_release_date.desc",
            "primary_release_date.gte": four_years, "primary_release_date.lte": today.isoformat(),
            "vote_count.gte": 3, "page": p}) for p in range(1, 7)],
        "recent": [("discover/movie", {
            "sort_by": "popularity.desc", "primary_release_date.gte": recent,
            "primary_release_date.lte": today.isoformat(), "vote_count.gte": 20, "page": p}) for p in range(1, 7)],
        "now_playing": [("movie/now_playing", {"region": "TH", "page": p}) for p in range(1, 4)],
    }
    for lang in ("ko", "ja", "zh"):
        src[lang] = [("discover/movie", {"with_original_language": lang, "sort_by": "popularity.desc",
                                         "vote_count.gte": 100, "page": p}) for p in range(1, 7)]
    for g in GENRES:
        src[f"g{g}"] = [("discover/movie", {"with_genres": str(g), "sort_by": "popularity.desc",
                                            "vote_count.gte": 200, "page": p}) for p in range(1, 6)]
    return src


def collect_ids(size):
    sources = source_jobs()
    flat = [(name, ep, prm) for name, jobs in sources.items() for ep, prm in jobs]
    print(f"🔎 กำลังรวบรวมรายชื่อหนังจาก {len(flat)} หน้า ...")
    with ThreadPoolExecutor(WORKERS) as ex:
        pages = list(ex.map(lambda j: tmdb_get(j[1], **j[2]) or {}, flat))
    per = {name: [] for name in sources}
    for (name, _ep, _prm), page in zip(flat, pages):
        per[name].extend(m["id"] for m in page.get("results", []) if m.get("id"))

    # ผสมแบบวนรอบ (round-robin) ให้ได้หนังหลากหลายแนว/ภาษา ไม่ถูกหนังดังจากแหล่งเดียวกินที่หมด
    iters = {n: iter(ids) for n, ids in per.items()}
    chosen, seen = [], set()
    while iters and len(chosen) < size:
        for n in list(iters):
            taken = 0
            for mid in iters[n]:
                if mid not in seen:
                    seen.add(mid)
                    chosen.append(mid)
                    taken += 1
                    if taken >= WEIGHTS.get(n, 1) or len(chosen) >= size:
                        break
            if taken == 0:
                del iters[n]
            if len(chosen) >= size:
                break
    return chosen


# ---------- 2) ดึงรายละเอียดหนังแต่ละเรื่อง ----------
def to_record(d):
    th = d.get("watch/providers", {}).get("results", {}).get("TH", {})
    flat = [p["provider_name"] for p in th.get("flatrate", [])]
    rent = []
    for p in th.get("rent", []) + th.get("buy", []):
        n = p["provider_name"]
        if n not in flat and n not in rent:
            rent.append(n)
    title_en, overview_en = "", ""
    for t in d.get("translations", {}).get("translations", []):
        if t.get("iso_639_1") == "en":
            data = t.get("data", {})
            title_en = data.get("title") or ""
            overview_en = data.get("overview") or ""
            break
    release = d.get("release_date") or ""
    genres = d.get("genres", [])
    return {
        "id": d["id"],
        "title": d.get("title") or d.get("original_title") or "",
        "original_title": d.get("original_title") or "",
        "title_en": title_en,
        "release_date": release,
        "year": release[:4],
        "popularity": d.get("popularity") or 0,
        "vote_average": d.get("vote_average") or 0,
        "vote_count": d.get("vote_count") or 0,
        "original_language": d.get("original_language") or "",
        "runtime": d.get("runtime") or 0,
        "genre_ids": [g["id"] for g in genres],
        "genres": [g["name"] for g in genres],
        "overview": d.get("overview") or overview_en,
        "providers": flat,
        "rent_buy": rent,
        "poster_path": d.get("poster_path"),
        "backdrop_path": d.get("backdrop_path"),
    }


def fetch_record(mid):
    d = tmdb_get(f"movie/{mid}", language="th-TH", append_to_response="watch/providers,translations")
    return to_record(d) if d and d.get("id") else None


def save_catalog(movies):
    offline.DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"built_at": datetime.now().isoformat(timespec="seconds"), "count": len(movies), "language": "th-TH"},
        "movies": movies,
    }
    tmp = offline.CATALOG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(offline.CATALOG_PATH)


def load_existing():
    try:
        data = json.loads(offline.CATALOG_PATH.read_text(encoding="utf-8"))
        return {m["id"]: m for m in data.get("movies", []) if m.get("id")}
    except Exception:
        return {}


# ---------- 3) ดาวน์โหลดรูป ----------
def download(job):
    url, path = job
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return True
            if r.status_code == 404:
                return False
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return False


def download_images(movies):
    offline.POSTER_DIR.mkdir(parents=True, exist_ok=True)
    offline.BACKDROP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{IMG}/w342{m['poster_path']}", offline.POSTER_DIR / f"{m['id']}.jpg")
            for m in movies if m.get("poster_path")]
    top = sorted(movies, key=lambda m: m.get("popularity") or 0, reverse=True)[:BACKDROP_TOP]
    jobs += [(f"{IMG}/w780{m['backdrop_path']}", offline.BACKDROP_DIR / f"{m['id']}.jpg")
             for m in top if m.get("backdrop_path")]
    jobs = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size > 0)]
    if not jobs:
        print("🖼️  รูปครบแล้ว ไม่มีอะไรต้องโหลดเพิ่ม")
        return
    print(f"🖼️  กำลังดาวน์โหลดรูป {len(jobs)} ไฟล์ ...")
    ok = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, success in enumerate(ex.map(download, jobs), start=1):
            ok += bool(success)
            if i % 200 == 0 or i == len(jobs):
                print(f"   {i}/{len(jobs)} (สำเร็จ {ok})")


# ---------- สถานะ ----------
def print_status():
    import llm_providers as L

    info = offline.catalog_info()
    print("— สถานะระบบออฟไลน์ —")
    if info["count"]:
        print(f"✅ ข้อมูลหนัง   : {info['count']:,} เรื่อง (สร้างเมื่อ {info['built_at'] or '-'}, โปสเตอร์ {info['posters']:,} ไฟล์)")
    else:
        print("❌ ข้อมูลหนัง   : ยังไม่มี → python build_offline_data.py")
    models = L.local_models()
    if models:
        print(f"✅ AI ในเครื่อง : Ollama พร้อม โมเดล: {', '.join(models)}")
    else:
        print("❌ AI ในเครื่อง : ไม่พบ Ollama → ติดตั้งจาก https://ollama.com แล้วรัน: ollama pull <ชื่อโมเดล>")
    if offline.whisper_ready():
        print(f"✅ ถอดเสียง     : Whisper '{offline.whisper_name()}' พร้อมใช้")
    elif not offline.whisper_installed():
        print("❌ ถอดเสียง     : ยังไม่ได้ติดตั้ง → pip install faster-whisper แล้วรัน --whisper small")
    else:
        print(f"❌ ถอดเสียง     : ยังไม่ได้ดาวน์โหลดโมเดล → python build_offline_data.py --skip-catalog --whisper {offline.whisper_name()}")
    cfg = offline.BASE_DIR / ".streamlit" / "config.toml"
    ok_cfg = cfg.exists() and "enableStaticServing" in cfg.read_text(encoding="utf-8", errors="ignore")
    print(("✅" if ok_cfg else "⚠️ ") + " รูปในเครื่อง   : " + (
        "เปิด enableStaticServing แล้ว" if ok_cfg else
        "ต้องใส่ enableStaticServing = true ใน .streamlit/config.toml (หัวข้อ [server]) ไม่งั้นรูปจะไม่ขึ้น"))


def main():
    ap = argparse.ArgumentParser(description="เตรียมข้อมูลสำหรับโหมดออฟไลน์ของ CineMatch")
    ap.add_argument("--size", type=int, default=2000, help="จำนวนหนังที่เก็บไว้ (ค่าเริ่มต้น 2000)")
    ap.add_argument("--no-images", action="store_true", help="ไม่ดาวน์โหลดรูป")
    ap.add_argument("--refresh", action="store_true", help="ดึงรายละเอียดใหม่ทั้งหมด (ไม่ใช้ข้อมูลเดิม)")
    ap.add_argument("--whisper", metavar="NAME", help="ดาวน์โหลดโมเดลถอดเสียง เช่น tiny, base, small, medium")
    ap.add_argument("--skip-catalog", action="store_true", help="ข้ามการดาวน์โหลดข้อมูลหนัง")
    ap.add_argument("--status", action="store_true", help="ตรวจสถานะแล้วจบ")
    args = ap.parse_args()

    if args.status:
        print_status()
        return

    if args.whisper:
        print(f"🎙️  กำลังดาวน์โหลดโมเดลถอดเสียง '{args.whisper}' (ครั้งเดียว อาจใช้เวลาหลายนาที) ...")
        try:
            offline.download_whisper(args.whisper)
            print(f"   เสร็จแล้ว — ตั้ง WHISPER_MODEL={args.whisper} ใน .env ถ้าไม่ใช่ small")
        except Exception as e:
            print(f"❌ ดาวน์โหลดโมเดลถอดเสียงไม่สำเร็จ: {e}")

    if args.skip_catalog:
        return
    if not KEY:
        sys.exit("❌ ไม่พบ TMDB_API_KEY ใน .env (ต้องใช้ตอนดาวน์โหลดข้อมูลครั้งแรกเท่านั้น)")

    existing = {} if args.refresh else load_existing()
    ids = collect_ids(args.size)
    todo = [i for i in ids if i not in existing]
    print(f"📚 ได้รายชื่อ {len(ids)} เรื่อง (มีอยู่แล้ว {len(ids) - len(todo)}, ต้องดึงเพิ่ม {len(todo)})")

    records = dict(existing)
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, rec in enumerate(ex.map(fetch_record, todo), start=1):
            if rec:
                records[rec["id"]] = rec
            if i % 100 == 0 or i == len(todo):
                print(f"   ดึงรายละเอียด {i}/{len(todo)}")
            if i % 400 == 0:
                save_catalog(list(records.values()))  # บันทึกระหว่างทาง กันงานหายถ้าโดนหยุดกลางคัน

    keep = set(ids) | set(existing)  # เก็บของเดิมไว้ด้วย ไม่ลบทิ้งเมื่อรันซ้ำด้วย --size น้อยกว่า
    movies = [records[i] for i in keep if i in records]
    save_catalog(movies)
    print(f"💾 บันทึก {len(movies):,} เรื่องแล้วที่ {offline.CATALOG_PATH}")

    if not args.no_images:
        download_images(movies)

    print("\n✅ เสร็จแล้ว ขั้นต่อไป")
    print("   1) ใส่ enableStaticServing = true ในหัวข้อ [server] ของ .streamlit/config.toml (ถ้ายังไม่ได้ใส่)")
    print("   2) ติดตั้ง Ollama + โมเดล (ดู README หรือรัน --status)")
    print("   3) ตั้ง OFFLINE_MODE=1 ใน .env หรือสลับ 'โหมดออฟไลน์' ในหน้า ⚙️ ตั้งค่า AI แล้วรีสตาร์ตแอป")


if __name__ == "__main__":
    main()