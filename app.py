import os
import json
import re
import html
import time
import requests
import streamlit as st
import offline
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from difflib import SequenceMatcher
from dotenv import load_dotenv
from google import genai
from google.genai import types

from llm_providers import (
    AllProvidersFailed,
    describe_chain,
    generate_text,
    groq_transcribe,
    has_any_llm,
    has_groq,
    model_options,
)

# ---------------------------------------------------------
# 1. Configuration & Initial Setup
# ---------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p"
GEMINI_MODEL = "gemini-3.8-flash"  # ใช้กับถอดเสียง (โมเดลแนะนำหนังตั้งใน llm_providers / .env)
# โมเดลถอดเสียงแยกจากโมเดลแนะนำหนัง โควตาฟรีนับแยกตามโมเดล จึงไม่แย่งกัน
TRANSCRIBE_MODELS = ["gemini-3.5-flash-lite", GEMINI_MODEL]  # ลองตัวแรกก่อน ถ้าใช้ไม่ได้ (404/โควตาหมด) ลองตัวถัดไป
NUM_RECS = 8  # จำนวนหนังที่ให้ AI แนะนำ (ปรับได้)
THAI_RECS = 3  # จำนวนหนังไทยขั้นต่ำที่อยากให้มีในผลแนะนำ (ใส่ 0 เพื่อปิด)
FUZZY_THRESHOLD = 0.75  # ความคล้ายขั้นต่ำของชื่อหนัง (0-1) ยิ่งต่ำยิ่งหาเจอง่ายแต่ผิดง่าย
AI_CACHE_TTL = 3600  # อายุแคชผล AI (วินาที) คำค้นเดิมภายในเวลานี้จะไม่เรียก AI ซ้ำ

# แพลตฟอร์มสำหรับตัวกรอง: ชื่อที่แสดง -> คำที่ใช้จับคู่กับชื่อ provider ของ TMDB (ตัวพิมพ์เล็ก)
PLATFORMS = {
    "Netflix": ("netflix",),
    "Disney+": ("disney",),
    "Apple TV+": ("apple tv",),
    "Prime Video": ("prime video", "amazon"),
    "Viu": ("viu",),
    "WeTV": ("wetv",),
    "iQIYI": ("iqiyi",),
}
MAX_MORE_ROUNDS = 3  # กด "ดูเพิ่ม" ได้กี่ครั้งต่อคำค้น

st.set_page_config(
    page_title="CineMatch AI | แนะนำหนังตรงใจคุณ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# 2. Custom CSS (Netflix-style theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+Thai:wght@300;400;600;700&display=swap');

    :root {
        --nf-red: #E50914;
        --nf-red-hover: #F6121D;
        --nf-black: #141414;
        --nf-surface: #181818;
        --nf-grey: #B3B3B3;
        --nf-text: #E5E5E5;
        --nf-green: #46D369;
        --side-pad: 4vw;
    }

    html { scroll-behavior: smooth; }
    html, body, .stApp {
        background-color: var(--nf-black);
        color: var(--nf-text);
        font-family: 'Noto Sans Thai', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"], #MainMenu, footer,
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

    .block-container {
        padding: 0 var(--side-pad) 3rem var(--side-pad) !important;
        max-width: 100% !important;
    }

    /* ---------- Top navigation ---------- */
    .nav {
        position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
        display: flex; align-items: center; gap: 1.2rem;
        padding: 14px var(--side-pad) 30px var(--side-pad);
        background: linear-gradient(180deg, rgba(20,20,20,.97) 0%, rgba(20,20,20,.88) 55%, rgba(20,20,20,0) 100%);
        pointer-events: none;
    }
    @media (max-width: 640px) { .nav-tag { display: none; } }
    .nav-logo {
        font-family: 'Bebas Neue', Impact, sans-serif;
        color: var(--nf-red);
        font-size: 2.1rem; letter-spacing: 2px; line-height: 1;
        text-shadow: 0 2px 8px rgba(0,0,0,.5);
    }
    .nav-tag { color: var(--nf-grey); font-size: .85rem; font-weight: 300; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        height: 62vh; min-height: 440px;
        margin: 0 calc(-1 * var(--side-pad));
        background-size: cover; background-position: center 20%;
        background-color: #000;
    }
    .hero::before {
        content: ""; position: absolute; inset: 0;
        background:
            linear-gradient(90deg, rgba(20,20,20,.92) 0%, rgba(20,20,20,.55) 40%, rgba(20,20,20,0) 75%),
            linear-gradient(180deg, rgba(20,20,20,0) 55%, var(--nf-black) 100%);
    }
    .hero-content {
        position: absolute; left: var(--side-pad); bottom: 24%;
        max-width: min(580px, 85vw);
        animation: heroIn .9s ease both;
    }
    @keyframes heroIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
    .hero-badge { color: var(--nf-grey); font-size: .95rem; margin-bottom: .4rem; }
    .hero-badge b { color: var(--nf-red); }
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.8rem); font-weight: 700; line-height: 1.1;
        color: #fff; margin: 0 0 .8rem 0;
        text-shadow: 2px 3px 12px rgba(0,0,0,.6);
    }
    .hero-meta { display: flex; flex-wrap: wrap; gap: .4rem 1rem; align-items: center; margin-bottom: .9rem; font-size: 1rem; }
    .score { color: var(--nf-green); font-weight: 700; }
    .hero-overview {
        color: #fff; font-size: 1.05rem; font-weight: 300; line-height: 1.6;
        display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
        text-shadow: 1px 1px 6px rgba(0,0,0,.7);
        margin-bottom: 1.2rem;
    }
    .hero-actions { display: flex; gap: .8rem; flex-wrap: wrap; }
    .hero-btn {
        display: inline-block; padding: .65rem 1.8rem; border-radius: 4px;
        background: rgba(109,109,110,.7); color: #fff !important; font-weight: 700;
        text-decoration: none !important; font-size: 1.05rem; transition: background .2s;
    }
    .hero-btn:hover { background: rgba(109,109,110,.45); }
    .hero-btn.primary { background: #fff; color: #000 !important; }
    .hero-btn.primary:hover { background: rgba(255,255,255,.75); }
    .prov-badge {
        display: inline-block; margin: 0 6px 6px 0; padding: 2px 10px;
        border: 1px solid #5a5a5a; border-radius: 3px;
        background: rgba(0,0,0,.55); color: #fff; font-size: .75rem;
    }

    /* ปุ่ม 👍/👎 ใต้การ์ด */
    .ai-card { margin-bottom: .4rem; }
    div[class*="st-key-like_"] div.stButton > button:first-child,
    div[class*="st-key-liked_"] div.stButton > button:first-child,
    div[class*="st-key-dislike_"] div.stButton > button:first-child {
        width: 100%; background: rgba(255,255,255,.06) !important; border: 1px solid #444 !important;
        color: var(--nf-text) !important; border-radius: 999px !important;
        font-size: .8rem !important; font-weight: 400 !important; padding: .25rem .4rem !important;
    }
    div[class*="st-key-like_"] div.stButton > button:first-child:hover,
    div[class*="st-key-dislike_"] div.stButton > button:first-child:hover {
        border-color: #fff !important; background: rgba(255,255,255,.12) !important;
    }
    div[class*="st-key-liked_"] div.stButton > button:first-child {
        background: rgba(70,211,105,.15) !important; border-color: var(--nf-green) !important;
        color: var(--nf-green) !important;
    }

    .prov-badge.rent { border-style: dashed; color: var(--nf-grey); background: transparent; }
    .rent-line { font-size: .75rem; color: var(--nf-grey); margin-top: 2px; }

    /* ---------- Search ---------- */
    .search-title { font-size: 1.5rem; font-weight: 700; color: #fff; margin: 1rem 0 .2rem 0; }
    .search-sub { color: var(--nf-grey); font-weight: 300; margin-bottom: .8rem; }

    /* ตัวบอกโมเดล AI (เหมือนที่ Claude แสดงชื่อโมเดล) */
    .model-badge {
        display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 8px;
        margin: .5rem 0 .8rem 0; padding: 4px 12px;
        border: 1px solid #3a3a3a; border-radius: 999px;
        background: rgba(255,255,255,.05); color: var(--nf-grey); font-size: .8rem;
    }
    .model-badge b { color: #fff; font-weight: 600; }
    .model-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--nf-green); box-shadow: 0 0 6px var(--nf-green);
    }
    .model-note { color: #f5a623; font-size: .75rem; }
    .model-note.ok { color: var(--nf-green); }
    .model-tag {
        border: 1px solid #5a5a5a; border-radius: 3px; padding: 0 6px;
        color: #fff; font-size: .7rem;
    }

    /* ตัวเลือกโมเดล AI (dropdown) */
    div[data-testid="stSelectbox"] label p { color: var(--nf-grey); font-size: .85rem; }
    div[data-baseweb="select"] > div {
        background-color: #2B2B2B; border: 1px solid #555; border-radius: 4px;
    }

    /* expander "ตั้งค่า AI" */
    [data-testid="stExpander"], [data-testid="stExpander"] details {
        border: none !important; background: transparent !important; margin: 0;
    }
    [data-testid="stExpander"] summary { color: var(--nf-grey); font-size: .85rem; padding-left: 0; }
    [data-testid="stExpander"] summary:hover { color: #fff; }

    /* Mood chips: เรียงชิดซ้าย ตัดบรรทัดเองเมื่อจอแคบ */
    .st-key-chips [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap; justify-content: flex-start; gap: .5rem;
    }
    .st-key-chips [data-testid="stColumn"],
    .st-key-chips [data-testid="column"] {
        width: auto !important; flex: 0 0 auto !important; min-width: 0 !important;
    }
    div[class*="st-key-chip_"] div.stButton > button:first-child {
        background: transparent !important; border: 1px solid #6d6d6e !important;
        color: var(--nf-text) !important; border-radius: 999px !important;
        font-size: .9rem !important; font-weight: 400 !important; padding: .3rem 1.1rem !important;
    }
    div[class*="st-key-chip_"] div.stButton > button:first-child:hover {
        border-color: #fff !important; background: rgba(255,255,255,.1) !important; transform: none;
    }

    /* Search panel (การ์ดโปร่งแสงซ้อนทับขอบล่างของ hero) */
    .st-key-searchpanel {
        position: relative; z-index: 5; margin-top: -2.5rem;
        padding: 1.2rem 1.6rem 1rem 1.6rem; border-radius: 14px;
        background: rgba(28,28,28,.78); border: 1px solid #2c2c2c;
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    }
    .st-key-searchpanel .search-title { margin-top: 0; }

    /* Tabs (พิมพ์ / พูด) → ปุ่มทรง pill */
    [data-baseweb="tab-list"] { gap: .5rem !important; border-bottom: none !important; background: transparent !important; }
    button[data-baseweb="tab"] {
        background: #262626 !important; border: 1px solid #3a3a3a !important; border-radius: 999px !important;
        height: auto !important; min-height: 0 !important; margin: 0 !important;
        padding: .45rem 1.3rem !important; color: var(--nf-grey) !important; font-weight: 600;
        transition: border-color .2s;
    }
    button[data-baseweb="tab"]:hover { border-color: #888 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background: #fff !important; border-color: #fff !important; }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] * { color: #000 !important; }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
        display: none !important; height: 0 !important; background: transparent !important;
    }

    /* ฟอร์มค้นหา (กด Enter เพื่อส่ง) — ซ่อนกรอบของ st.form */
    [data-testid="stForm"] { border: none !important; padding: 0 !important; }

    /* ช่องพิมพ์ทรงแคปซูล */
    [data-testid="stTextInput"] div[data-baseweb="input"],
    [data-testid="stTextInput"] div[data-baseweb="base-input"] {
        background-color: #1f1f1f !important; border-radius: 999px !important;
    }
    [data-testid="stTextInput"] div[data-baseweb="input"] {
        border: 1px solid #444 !important; min-height: 3.2rem; padding-left: .7rem;
        transition: border-color .2s, box-shadow .2s;
    }
    [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: var(--nf-red) !important; box-shadow: 0 0 0 3px rgba(229,9,20,.25);
    }
    [data-testid="stTextInput"] input {
        background-color: transparent !important; color: #fff !important;
        font-size: 1.05rem; padding: .8rem .9rem;
    }
    [data-testid="stTextInput"] input::placeholder { color: #8c8c8c; }

    /* ตัวอัดเสียง */
    [data-testid="stAudioInput"] > div {
        background: #1f1f1f; border: 1px solid #444; border-radius: 999px;
    }
    .voice-label { color: var(--nf-grey); font-size: .85rem; margin: .8rem 0 .3rem .4rem; }

    /* ปุ่มส่งในฟอร์ม: เต็มความกว้างคอลัมน์ ทรงแคปซูล */
    [data-testid="stFormSubmitButton"],
    [data-testid="stFormSubmitButton"] > button { width: 100% !important; }
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 999px !important; padding: .75rem 1rem !important;
    }

    /* ปุ่มล้างผลลัพธ์ (แบบ ghost) */
    div[class*="st-key-clear_results"] div.stButton > button:first-child,
    div[class*="st-key-clear_prefs"] div.stButton > button:first-child,
    div[class*="st-key-more_btn"] div.stButton > button:first-child {
        background: transparent !important; border: 1px solid #555 !important; color: var(--nf-grey) !important;
        border-radius: 999px !important; font-size: .85rem !important; font-weight: 400 !important;
        padding: .25rem .9rem !important;
    }
    div[class*="st-key-clear_results"] div.stButton > button:first-child:hover,
    div[class*="st-key-clear_prefs"] div.stButton > button:first-child:hover,
    div[class*="st-key-more_btn"] div.stButton > button:first-child:hover {
        border-color: #fff !important; color: #fff !important; background: rgba(255,255,255,.08) !important;
    }

    /* ตัวกรองแพลตฟอร์ม */
    div[data-testid="stPills"] label p, div[data-testid="stPills"] p { font-size: .85rem; }
    div[data-testid="stPills"] label p { color: var(--nf-grey); }

    div.stButton > button:first-child,
    div[data-testid="stFormSubmitButton"] > button:first-child {
        background-color: var(--nf-red); color: #fff;
        font-size: 1.05rem; font-weight: 700;
        border: none; border-radius: 4px; padding: .6rem 1.8rem;
        transition: background-color .2s;
    }
    div.stButton > button:first-child:hover,
    div[data-testid="stFormSubmitButton"] > button:first-child:hover {
        background-color: var(--nf-red-hover); color: #fff;
    }
    div.stButton > button:first-child:focus-visible,
    div[data-testid="stFormSubmitButton"] > button:first-child:focus-visible {
        outline: 2px solid #fff; outline-offset: 2px;
    }

    /* ---------- Rows & cards ---------- */
    .row-title {
        font-size: 1.35rem; font-weight: 700; color: var(--nf-text);
        margin: 1.6rem 0 0 0;
    }
    .row {
        display: flex; gap: 10px; overflow-x: auto; scroll-snap-type: x proximity;
        padding: 34px var(--side-pad);
        margin: 0 calc(-1 * var(--side-pad));
        scrollbar-width: thin; scrollbar-color: #444 transparent;
    }
    .row::-webkit-scrollbar { height: 6px; }
    .row::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }

    .card {
        position: relative; flex: 0 0 auto; scroll-snap-align: start;
        width: clamp(130px, 14vw, 200px); aspect-ratio: 2 / 3;
        border-radius: 4px; overflow: hidden; background: var(--nf-surface);
        transition: transform .25s ease, box-shadow .25s ease;
        text-decoration: none !important; color: inherit !important;
    }
    .card:hover, .card:focus-visible {
        transform: scale(1.14); z-index: 10;
        box-shadow: 0 10px 30px rgba(0,0,0,.8);
        outline: none;
    }
    .card img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .card-info {
        position: absolute; left: 0; right: 0; bottom: 0;
        padding: 40px 10px 10px 10px;
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,.92) 45%);
        opacity: 0; transition: opacity .25s ease;
    }
    .card:hover .card-info, .card:focus-visible .card-info { opacity: 1; }
    .card-title { font-size: .85rem; font-weight: 700; color: #fff; line-height: 1.25; }
    .card-meta { font-size: .72rem; color: var(--nf-grey); margin: 3px 0; display: flex; gap: 8px; }
    .card-ov {
        font-size: .68rem; color: #d2d2d2; line-height: 1.35; margin-bottom: 4px;
        display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
    }
    .card .prov-badge { font-size: .62rem; padding: 1px 6px; margin: 0 4px 0 0; }
    @media (hover: none) { .card-info { opacity: 1; } .card-ov { display: none; } }

    /* Top 10 (ตัวเลขใหญ่ข้างโปสเตอร์) */
    .top-item { display: flex; align-items: flex-end; flex: 0 0 auto; scroll-snap-align: start; margin-right: 22px; }
    .top-num {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: clamp(6rem, 9vw, 8.5rem); line-height: .8;
        color: var(--nf-black); -webkit-text-stroke: 3px #7a7a7a;
        margin-right: -.1em; user-select: none;
    }
    .top-item .card { width: clamp(110px, 11vw, 160px); z-index: 1; }
    .top-item .card:hover { z-index: 10; }

    /* ---------- AI recommendations ---------- */
    .ai-grid {
        display: grid; gap: 22px;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        margin: 1rem 0 2rem 0;
    }
    .ai-card { background: var(--nf-surface); border-radius: 8px; overflow: hidden; transition: transform .2s ease, box-shadow .2s ease; }
    .ai-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,0,0,.65); }
    .ai-card img { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; display: block; }
    .ai-body { padding: 12px 14px 16px 14px; }
    .ai-title { font-weight: 700; color: #fff; font-size: 1rem; line-height: 1.3; }
    .ai-meta { display: flex; flex-wrap: wrap; gap: 2px 10px; font-size: .8rem; color: var(--nf-grey); margin: 4px 0 8px 0; }
    .ai-genres { font-size: .78rem; color: var(--nf-grey); margin-bottom: 8px; }
    .ai-reason {
        border-left: 3px solid var(--nf-red); padding-left: 10px;
        font-size: .85rem; color: #d2d2d2; line-height: 1.5; margin-bottom: 10px;
    }
    .ai-link {
        display: inline-block; margin-top: 4px; padding: 4px 12px; border-radius: 3px;
        background: rgba(255,255,255,.12); color: #fff !important; font-size: .8rem;
        text-decoration: none !important; transition: background .2s;
    }
    .ai-link:hover { background: rgba(255,255,255,.25); }
    .ai-head { font-size: 1.5rem; font-weight: 700; color: #fff; margin: 1.5rem 0 0 0; }
    .ai-head span { color: var(--nf-grey); font-weight: 300; }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center; color: #757575; font-size: .8rem; line-height: 1.8;
        border-top: 1px solid #2a2a2a; margin-top: 2.5rem; padding-top: 1.5rem;
    }

    @media (prefers-reduced-motion: reduce) {
        .card, .card-info, .hero-btn, .hero-content, .ai-card { transition: none; animation: none; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. TMDB API Helper Functions
# ---------------------------------------------------------
PLACEHOLDER_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='500' height='750'>"
    "<rect width='100%' height='100%' fill='%23222'/>"
    "<text x='50%' y='50%' fill='%23777' font-size='32' text-anchor='middle' "
    "font-family='Arial'>No Poster</text></svg>"
)


def fetch_tmdb_movies(endpoint, params=None):
    if offline.use_local(TMDB_API_KEY):
        return offline.fetch(endpoint, params)
    if not TMDB_API_KEY:
        return []
    default_params = {"api_key": TMDB_API_KEY, "language": "th-TH", "page": 1}
    if params:
        default_params.update(params)
    try:
        res = requests.get(f"{TMDB_BASE_URL}/{endpoint}", params=default_params, timeout=10)
        if res.status_code == 200:
            return res.json().get("results", [])
    except requests.RequestException:
        # เน็ตหลุด/ติดต่อ TMDB ไม่ได้: พักการเรียกชั่วคราว แล้วใช้ข้อมูลในเครื่องแทน (ถ้ามี)
        offline.mark_network_down()
        if offline.catalog_available():
            return offline.fetch(endpoint, params)
    except Exception:
        pass
    return []


def get_movie_details(movie_id):
    if offline.use_local(TMDB_API_KEY):
        return offline.details(movie_id)
    if not TMDB_API_KEY:
        return None
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "th-TH",
            "append_to_response": "watch/providers,videos",
            "include_video_language": "th,en",
        }
        res = requests.get(url, params=params, timeout=10).json()
        th = res.get("watch/providers", {}).get("results", {}).get("TH", {})
        provider_names = [p["provider_name"] for p in th.get("flatrate", [])]  # สตรีมแบบสมาชิก
        rent_buy = []  # เช่า/ซื้อ (ไม่นับเจ้าที่มีสตรีมอยู่แล้ว)
        for p in th.get("rent", []) + th.get("buy", []):
            n = p["provider_name"]
            if n not in provider_names and n not in rent_buy:
                rent_buy.append(n)

        # ตัวอย่างหนัง (YouTube) — เลือก Trailer ก่อน ถ้าไม่มีใช้คลิปอื่นแทน
        videos = [v for v in res.get("videos", {}).get("results", []) if v.get("site") == "YouTube"]
        trailer = next((v for v in videos if v.get("type") == "Trailer"), videos[0] if videos else None)

        return {
            "id": res.get("id"),
            "title": res.get("title") or res.get("original_title") or "ไม่ทราบชื่อเรื่อง",
            "overview": res.get("overview") or "ไม่มีข้อมูลเรื่องย่อภาษาไทย",
            "rating": round(res.get("vote_average", 0) or 0, 1),
            "release_date": (res.get("release_date") or "N/A")[:4],
            "runtime": res.get("runtime") or 0,
            "genres": [g["name"] for g in res.get("genres", [])][:3],
            "poster": f"{IMG_BASE}/w500{res['poster_path']}" if res.get("poster_path") else None,
            "backdrop": f"{IMG_BASE}/w1280{res['backdrop_path']}" if res.get("backdrop_path") else None,
            "trailer": f"https://www.youtube.com/watch?v={trailer['key']}" if trailer else None,
            "providers": provider_names,
            "rent_buy": rent_buy,
            "language": res.get("original_language"),
        }
    except requests.RequestException:
        offline.mark_network_down()
        return offline.details(movie_id) if offline.catalog_available() else None
    except Exception:
        return None


def search_movie_by_title(title, year=None, thai_title=None):
    """ค้นหาหนังใน TMDB: ลองชื่อ+ปี ก่อน แล้วค่อยลองชื่ออย่างเดียว และชื่อไทย"""
    if offline.use_local(TMDB_API_KEY):
        mid = offline.resolve_title(title, year, thai_title)
        return get_movie_details(mid) if mid else None
    attempts = []
    if title and year:
        attempts.append({"query": title, "year": year})
    if title:
        attempts.append({"query": title})
    if thai_title and thai_title != title:
        attempts.append({"query": thai_title})
    for params in attempts:
        results = fetch_tmdb_movies("search/movie", params)
        if results:
            return get_movie_details(results[0]["id"])
    return None


GENRE_NAMES = {
    28: "แอคชั่น", 12: "ผจญภัย", 16: "แอนิเมชัน", 35: "ตลก", 80: "อาชญากรรม",
    99: "สารคดี", 18: "ดราม่า", 10751: "ครอบครัว", 14: "แฟนตาซี", 36: "ประวัติศาสตร์",
    27: "สยองขวัญ", 10402: "ดนตรี", 9648: "ลึกลับ", 10749: "โรแมนติก",
    878: "ไซไฟ", 53: "ระทึกขวัญ", 10752: "สงคราม", 37: "คาวบอย",
}


def norm_text(text):
    """ทำให้ข้อความเทียบกันง่าย (ตัดช่องว่าง/เครื่องหมายวรรคตอน ไม่แตะสระและวรรณยุกต์ไทย)"""
    return re.sub(r"[\s\-_:.,!?'\"“”()]+", "", (text or "").lower())


# ---------- ค้นหาชื่อใกล้เคียง (fuzzy) ----------
# ตัวอักษรไทยที่ออกเสียงคล้ายกัน ระบบถอดเสียงมักสลับกัน เช่น ธี่หยด -> ที่หยด
CONFUSABLE_GROUPS = ["ทธฑฒถ", "สศษซ", "พภ", "คขฆ", "ชฉ", "ดฎ", "ตฏ", "นณ", "ญย", "ฟฝ", "ไใ"]
_CONF_MAP = {ch: g[0] for g in CONFUSABLE_GROUPS for ch in g}
_THAI_MARKS = "[\u0e47-\u0e4c]"  # ไม้ไต่คู้ วรรณยุกต์ ทัณฑฆาต


def phonetic_key(text):
    """แปลงข้อความเป็น 'รหัสเสียง': ตัดวรรณยุกต์ และรวมพยัญชนะที่เสียงคล้ายกันให้เป็นตัวเดียว"""
    t = re.sub(_THAI_MARKS, "", norm_text(text))
    return "".join(_CONF_MAP.get(c, c) for c in t)


# ให้ระบบออฟไลน์ค้นชื่อหนังแบบทนสะกดผิด/ถอดเสียงผิดด้วยรหัสเสียงเดียวกัน
offline.set_keyfunc(phonetic_key)


def similarity(a, b):
    """ความคล้ายของชื่อ 0-1 (1 = ตรงหรือเป็นส่วนหนึ่งของกันและกัน)"""
    ka, kb = phonetic_key(a), phonetic_key(b)
    if len(ka) < 3 or len(kb) < 3:
        return 0.0
    if ka in kb or (len(kb) >= 4 and kb in ka):
        return 1.0
    return SequenceMatcher(None, ka, kb).ratio()


def query_variants(query, max_variants=12):
    """สร้างคำค้นสำรองโดยสลับตัวอักษรที่ออกเสียงคล้ายกันทีละตัว (ส่งไปค้น TMDB ด้วย)"""
    q = re.sub(r"\s+", " ", (query or "").strip())
    if not q or len(q) > 25:
        return [q] if q else []
    groups = CONFUSABLE_GROUPS + ["\u0e48\u0e49"]  # สลับ ่ ้ ด้วย
    variants = [q]
    for i, ch in enumerate(q):
        for g in groups:
            if ch in g:
                variants += [q[:i] + alt + q[i + 1:] for alt in g if alt != ch]
    return list(dict.fromkeys(variants))[:max_variants]


def friendly_error(e):
    if isinstance(e, AllProvidersFailed):
        return str(e)
    s = str(e)
    if "429" in s or "RESOURCE_EXHAUSTED" in s:
        return "โควตา Gemini ของวันนี้หมดแล้ว (ลองใหม่ภายหลัง หรือเปิด billing / เปลี่ยนโมเดล)"
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def load_thai_pool():
    """รวบรวมรายชื่อหนังไทยจาก TMDB (ยอดนิยม + หนังผี + หนังออกใหม่ 4 ปีล่าสุด)
    ไว้ให้ AI เลือก และใช้เทียบชื่อแบบ fuzzy ตอนค้นหา"""
    today = date.today()
    since = (today - timedelta(days=365 * 4)).isoformat()
    bases = [
        {"sort_by": "popularity.desc", "vote_count.gte": 10},
        {"sort_by": "popularity.desc", "with_genres": "27", "vote_count.gte": 5},
        {
            "sort_by": "primary_release_date.desc",
            "primary_release_date.gte": since,
            "primary_release_date.lte": today.isoformat(),
            "vote_count.gte": 5,
        },
    ]
    param_list = [
        {"with_original_language": "th", "region": "TH", "page": page, **base}
        for base in bases for page in (1, 2, 3)
    ]
    with ThreadPoolExecutor(max_workers=8) as ex:
        pages = list(ex.map(lambda prm: fetch_tmdb_movies("discover/movie", prm), param_list))

    pool = {}
    for page in pages:
        for m in page:
            if m["id"] in pool:
                continue
            pool[m["id"]] = {
                "id": m["id"],
                "title": m.get("title") or m.get("original_title") or "",
                "original_title": m.get("original_title") or "",
                "year": (m.get("release_date") or "")[:4],
                "genres": ", ".join(GENRE_NAMES[g] for g in m.get("genre_ids", []) if g in GENRE_NAMES),
            }
    return list(pool.values())[:150]


def search_direct(query, limit=3):
    """ค้นชื่อหนังตรงๆ แบบทนต่อการสะกดผิด/ถอดเสียงผิด
    1) ค้น TMDB ด้วยคำค้นเดิม + คำค้นที่สลับตัวอักษรเสียงคล้าย
    2) เทียบกับรายชื่อหนังไทยที่โหลดไว้ด้วยรหัสเสียง
    แล้วจัดอันดับตามความคล้ายของชื่อ"""
    if len(norm_text(query)) < 3:
        return []

    variants = query_variants(query)
    jobs = [(variants[0], "th-TH"), (variants[0], "en-US")] + [(v, "th-TH") for v in variants[1:]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        pages = list(ex.map(
            lambda j: fetch_tmdb_movies("search/movie", {"query": j[0], "language": j[1]}), jobs
        ))

    cands = {}  # id -> (score, popularity)
    for page in pages:
        for m in page:
            names = [n for n in (m.get("title"), m.get("original_title")) if n]
            score = max((similarity(query, n) for n in names), default=0.0)
            if score >= FUZZY_THRESHOLD:
                old = cands.get(m["id"], (0.0, 0.0))
                cands[m["id"]] = (max(old[0], score), m.get("popularity", 0) or 0)

    try:
        for p in load_thai_pool():
            score = max(similarity(query, p["title"]), similarity(query, p.get("original_title", "")))
            if score >= FUZZY_THRESHOLD and p["id"] not in cands:
                cands[p["id"]] = (score, 0)
    except Exception:
        pass

    ranked = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    if not ranked:
        return []
    with ThreadPoolExecutor(max_workers=len(ranked)) as ex:
        details = list(ex.map(get_movie_details, [mid for mid, _ in ranked]))
    out = []
    for (_, (score, _pop)), d in zip(ranked, details):
        if d and d.get("id"):
            d["match_score"] = score
            out.append(d)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_row(endpoint, params_items=(), limit=12):
    """ดึงรายชื่อหนัง + รายละเอียดแบบขนาน และแคชไว้ 1 ชั่วโมง"""
    raw = fetch_tmdb_movies(endpoint, dict(params_items))
    ids = [m["id"] for m in raw[:limit]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        details = list(ex.map(get_movie_details, ids))
    return [d for d in details if d]


# ---------------------------------------------------------
# 4. AI Helper Functions (ถอดเสียง + แนะนำหนัง)
#    การแนะนำหนังเรียกผ่าน llm_providers: Gemini -> Groq -> OpenRouter -> Mistral
#    - ผู้ใช้เลือกโมเดลเองได้ (preferred) ถ้าตัวที่เลือกใช้ไม่ได้จะสลับตามลำดับสำรองอัตโนมัติ
#    - ถ้าไม่เลือก = โหมดอัตโนมัติ สลับเมื่อเจ้าที่ใช้อยู่โควตาหมด/ใช้ไม่ได้
# ---------------------------------------------------------
def _is_gemini_retryable(s):
    return "404" in s or "NOT_FOUND" in s or "429" in s or "RESOURCE_EXHAUSTED" in s


def transcribe_audio(audio_bytes, mime_type="audio/wav"):
    """ถอดเสียงเป็นข้อความ: ลอง Gemini ก่อน ถ้าใช้ไม่ได้สลับไป Whisper บน Groq หรือ Whisper ในเครื่อง"""
    # ออฟไลน์ หรือไม่มีคีย์ออนไลน์เลย: ใช้ Whisper ในเครื่อง
    if offline.forced() or not (GEMINI_API_KEY or has_groq()):
        if offline.whisper_ready():
            try:
                local_hint = "ชื่อหนัง: " + ", ".join(p["title"] for p in load_thai_pool()[:25] if p["title"])
            except Exception:
                local_hint = ""
            return offline.local_transcribe(audio_bytes, local_hint)
        if offline.forced():
            raise RuntimeError("ยังไม่มีโมเดลถอดเสียงในเครื่อง (รัน python build_offline_data.py --whisper small)")

    hint = ""
    try:
        titles = [p["title"] for p in load_thai_pool() if p["title"]][:100]
        if titles:
            hint = (
                "\nถ้าผู้พูดเอ่ยชื่อหนังไทย ให้สะกดตามรายชื่อนี้ถ้าเสียงใกล้เคียง "
                "(ใช้เป็นแนวทางสะกดเท่านั้น ห้ามใส่ชื่อที่ไม่ได้พูด): " + ", ".join(titles)
            )
    except Exception:
        pass

    text, last_error = None, None

    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        contents = [
            "ถอดเสียงพูดในไฟล์นี้เป็นข้อความตามที่ได้ยิน (ภาษาไทยหรืออังกฤษ) "
            "ตอบเฉพาะข้อความที่ถอดได้ ไม่ต้องอธิบายเพิ่ม "
            "ถ้าไม่มีเสียงพูดให้ตอบว่าว่างเปล่า" + hint,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ]
        for model_name in TRANSCRIBE_MODELS:
            try:
                response = client.models.generate_content(model=model_name, contents=contents)
                text = (response.text or "").strip()
                break
            except Exception as e:
                last_error = e
                # โมเดลหาย/โควตาหมด -> ลองตัวถัดไป; error อื่นถ้ามี Groq หรือ Whisper ในเครื่องสำรอง ก็ไปใช้ตัวสำรอง
                if not _is_gemini_retryable(str(e)) and not has_groq() and not offline.whisper_ready():
                    raise

    if text is None:
        if has_groq():
            # รายชื่อหนังสำหรับช่วยสะกด (Whisper รับ prompt สั้นๆ)
            short_hint = ""
            try:
                short_hint = "ชื่อหนัง: " + ", ".join(p["title"] for p in load_thai_pool()[:25] if p["title"])
            except Exception:
                pass
            text = groq_transcribe(audio_bytes, mime_type, short_hint)
        elif offline.whisper_ready():
            text = offline.local_transcribe(audio_bytes, "")
        else:
            raise last_error or RuntimeError("ไม่มีบริการถอดเสียงที่ใช้ได้")

    return "" if text in ("ว่างเปล่า", "") else text


@st.cache_data(ttl=3600, show_spinner=False)
def load_new_pool():
    """หนังใหม่ที่เพิ่งฉาย/ออกใน 4 เดือนล่าสุด (ทุกภาษา) ไว้ให้ AI รู้จักหนังที่เกินความรู้ของโมเดล"""
    today = date.today()
    since = (today - timedelta(days=120)).isoformat()
    base = {
        "sort_by": "popularity.desc",
        "primary_release_date.gte": since,
        "primary_release_date.lte": today.isoformat(),
        "vote_count.gte": 20,
    }
    jobs = [
        ("movie/now_playing", {"region": "TH", "page": 1}),
        ("discover/movie", {**base, "page": 1}),
        ("discover/movie", {**base, "page": 2}),
    ]
    with ThreadPoolExecutor(max_workers=3) as ex:
        pages = list(ex.map(lambda j: fetch_tmdb_movies(j[0], j[1]), jobs))

    pool = {}
    for page in pages:
        for m in page:
            if m["id"] in pool:
                continue
            pool[m["id"]] = {
                "id": m["id"],
                "title": m.get("title") or m.get("original_title") or "",
                "year": (m.get("release_date") or "")[:4],
                "genres": ", ".join(GENRE_NAMES[g] for g in m.get("genre_ids", []) if g in GENRE_NAMES),
            }
    return list(pool.values())[:40]


def ai_recommend(query, preferred=None, exclude_movies=None, include_direct=True, stats=None,
                 liked=None, disliked=None):
    """ค้นชื่อตรงๆ (fuzzy) + ให้ AI เลือกหนัง NUM_RECS เรื่อง
    preferred = (provider_key, model) ที่ผู้ใช้เลือก หรือ None = อัตโนมัติ
    exclude_movies = หนังที่แสดงไปแล้ว (ใช้ตอนกด "ดูเพิ่ม" จะไม่แนะนำซ้ำ)
    include_direct = False เพื่อข้ามการค้นชื่อตรงๆ (ใช้ตอน "ดูเพิ่ม" และตอนทดสอบ AI ล้วนๆ)
    stats = dict ที่ถ้าส่งมาจะถูกเติม proposed (AI เสนอกี่เรื่อง) / resolved (หาเจอใน TMDB กี่เรื่อง)
    คืนค่า (movies, ai_error, ai_info)
    - ai_info = {"provider", "model", "skipped", "chosen"} ของโมเดลที่ตอบ (None ถ้า AI ใช้ไม่ได้)
    - ถ้า AI ใช้ไม่ได้ทุกเจ้า จะยังได้ผลจากการค้นชื่อ (ถ้า include_direct)"""
    exclude_movies = exclude_movies or []
    exclude_ids = {m["id"] for m in exclude_movies}
    exclude_titles = [m["title"] for m in exclude_movies][:40]

    # รสนิยมที่ผู้ใช้กด 👍/👎 ไว้: ส่งให้ AI เป็นบริบท และไม่แนะนำเรื่องเหล่านี้ซ้ำ
    liked, disliked = liked or [], disliked or []
    avoid_ids = {m["id"] for m in liked + disliked}
    taste_rule = ""
    if liked or disliked:
        taste_rule = "- ให้ความสำคัญกับคำขอของผู้ใช้ก่อน ส่วนรสนิยมด้านล่างใช้ช่วยเลือกระหว่างตัวเลือกที่เข้ากับคำขอเท่านั้น\n"
    if liked:
        taste_rule += (
            "    - ผู้ใช้เคยกดชอบหนังเหล่านี้ ใช้เป็นแนวทางรสนิยม (แนวเรื่อง โทน สไตล์) "
            "แต่ไม่ต้องแนะนำเรื่องเดิมซ้ำ: " + ", ".join(m["title"] for m in liked[:15]) + "\n"
        )
    if disliked:
        taste_rule += (
            "    - ผู้ใช้กดไม่ชอบหนังเหล่านี้ ห้ามแนะนำ และเลี่ยงหนังที่คล้ายกันมาก: "
            + ", ".join(m["title"] for m in disliked[:15])
        )

    local_mode = offline.use_local(TMDB_API_KEY)
    if local_mode:
        # ออฟไลน์: คัดรายชื่อจากข้อมูลในเครื่องให้ AI เลือก (ตัดเรื่องที่แสดงไปแล้วออก)
        cand = [p for p in offline.candidates(query, n=30, thai_min=THAI_RECS) if p["id"] not in exclude_ids]
        pool, fresh = cand, []
        known_ids = {p["id"] for p in cand}
        pool_text = "\n".join(f"{p['id']} | {p['title']} | {p['year']} | {p['genres']}" for p in cand)
        new_text = ""
    else:
        full_pool = load_thai_pool() if THAI_RECS > 0 else []
        try:
            new_pool = load_new_pool()
        except Exception:
            new_pool = []
        # id ที่ AI อ้างได้โดยไม่ต้องค้นชื่อ (ใช้ตรวจว่า tmdb_id ที่ AI ใส่มามาจากรายชื่อของเราจริง)
        known_ids = {p["id"] for p in full_pool} | {p["id"] for p in new_pool}

        pool = [p for p in full_pool if p["id"] not in exclude_ids]
        fresh = [p for p in new_pool if p["id"] not in exclude_ids]
        pool_text = "\n".join(f"{p['id']} | {p['title']} | {p['year']} | {p['genres']}" for p in pool)
        new_text = "\n".join(f"{p['id']} | {p['title']} | {p['year']} | {p['genres']}" for p in fresh)

    thai_rule = ""
    if THAI_RECS > 0:
        thai_rule = (
            f"- ต้องมีหนังไทย (สร้างในประเทศไทย) อย่างน้อย {THAI_RECS} เรื่อง "
            "ที่เข้ากับสิ่งที่ผู้ใช้ต้องการ เว้นแต่ผู้ใช้ระบุชาติ/ประเทศของหนังไว้ชัดเจนแล้ว\n"
            "    - สำหรับหนังไทย ให้เลือกจากรายชื่อด้านล่างก่อน และใส่ tmdb_id ตามรายการ "
            "ถ้าไม่มีเรื่องที่เหมาะในรายการค่อยใช้ความรู้ของคุณ แล้วใส่ tmdb_id เป็น null\n"
            f"    รายชื่อหนังไทยจาก TMDB (รูปแบบ id | ชื่อ | ปี | แนว):\n{pool_text}"
        )

    new_rule = ""
    if new_text:
        new_rule = (
            "- หนังที่เพิ่งออกฉายในช่วง 4 เดือนที่ผ่านมามีรายชื่อด้านล่าง (ความรู้ของคุณอาจยังไม่ครอบคลุม) "
            "ถ้ามีเรื่องที่เข้ากับสิ่งที่ผู้ใช้ต้องการ ให้เลือกมา 1-2 เรื่องพร้อมใส่ tmdb_id ตามรายการ "
            "ถ้าไม่เข้ากับคำขอไม่ต้องเลือก\n"
            f"    รายชื่อหนังใหม่ (รูปแบบ id | ชื่อ | ปี | แนว):\n{new_text}"
        )

    if local_mode:
        # รายชื่อออฟไลน์มีหลายแนว/หลายภาษา ไม่ใช่เฉพาะหนังไทย จึงใช้คำสั่งนี้แทน thai_rule
        thai_rule = (
            "- เลือกหนังจากรายชื่อด้านล่างเท่านั้น และใส่ tmdb_id ตามรายการ\n"
            + (f"    - ต้องมีหนังไทยอย่างน้อย {THAI_RECS} เรื่อง ถ้าในรายชื่อมี "
               "เว้นแต่ผู้ใช้ระบุชาติ/ประเทศของหนังไว้ชัดเจนแล้ว\n" if THAI_RECS > 0 else "")
            + f"    รายชื่อหนัง (รูปแบบ id | ชื่อ | ปี | แนว):\n{pool_text}"
        )

    exclude_rule = ""
    if exclude_titles:
        exclude_rule = "- ห้ามแนะนำเรื่องเหล่านี้ซ้ำ เพราะแนะนำไปแล้ว: " + ", ".join(exclude_titles)

    prompt_text = f"""
    คุณคือ AI แนะนำภาพยนตร์
    ผู้ใช้ต้องการดูหนังตามนี้: '{query}'
    เลือกหนังที่ตรงที่สุดมา {NUM_RECS} เรื่อง โดยมีเงื่อนไขดังนี้
    {thai_rule}
    {new_rule}
    {exclude_rule}
    {taste_rule}
    - ต้องเป็นหนังที่มีอยู่จริงเท่านั้น ห้ามแต่งชื่อเรื่อง
    - ผสมทั้งหนังดังและหนังที่คนอาจไม่เคยได้ยิน
    ตอบกลับเป็น JSON Array เท่านั้น ไม่ต้องมีคำอธิบายหรือ markdown โครงสร้าง:
    [{{"title": "ชื่อสากล/ชื่ออังกฤษตามที่ใช้ใน TMDB (หนังไทยให้ใช้ชื่ออังกฤษหรือ romanized เช่น Shutter, Bad Genius)",
       "thai_title": "ชื่อไทย (ถ้าเป็นหนังไทย ไม่ใช่ให้ใส่ค่าว่าง)",
       "year": ปีที่ออกฉายเป็นตัวเลข,
       "tmdb_id": id จากรายชื่อด้านบน หรือ null,
       "reason": "เหตุผลสั้นๆ ภาษาไทย"}}]
    """

    def resolve(rec):
        # 1) ใช้ id จาก TMDB ถ้า AI เลือกจากรายชื่อที่เราให้ไว้
        try:
            tid = int(rec.get("tmdb_id"))
        except (TypeError, ValueError):
            tid = None
        if tid in known_ids:
            details = get_movie_details(tid)
            if details and details.get("id"):
                return details
        # 2) ไม่มี id ก็ค้นจากชื่อ/ปี/ชื่อไทย
        return search_movie_by_title(rec.get("title", ""), rec.get("year"), rec.get("thai_title"))

    recs, ai_error, ai_info = [], None, None
    with ThreadPoolExecutor(max_workers=NUM_RECS + 2) as ex:
        # ค้นชื่อตรงๆ ไปพร้อมกับที่รอ AI
        direct_future = ex.submit(search_direct, query) if include_direct else None
        try:
            # ลองโมเดลที่ผู้ใช้เลือกก่อน (ถ้ามี) แล้วสลับเจ้าอัตโนมัติอยู่ข้างใน
            text, ai_info = generate_text(prompt_text, preferred=preferred)
            json_match = re.search(r"\[.*\]", text or "", re.DOTALL)
            if not json_match:
                raise ValueError("AI ตอบกลับในรูปแบบที่อ่านไม่ได้ ลองค้นหาอีกครั้ง")
            recs = [r for r in json.loads(json_match.group(0)) if isinstance(r, dict)]
        except Exception as e:
            ai_error, ai_info = e, None
        found = list(ex.map(resolve, recs))
        direct = direct_future.result() if direct_future else []

    if stats is not None:
        stats["proposed"] = len(recs)
        stats["resolved"] = sum(1 for d in found if d and d.get("id"))

    if ai_error and not direct:
        raise ai_error

    movies, seen = [], set(exclude_ids)
    # หนังที่ชื่อตรง/ใกล้เคียงกับคำค้น ขึ้นก่อนเสมอ
    for details in direct:
        if details["id"] not in seen:
            seen.add(details["id"])
            exact = details.get("match_score", 1.0) >= 0.999
            details["ai_reason"] = "ตรงกับชื่อที่คุณค้นหา" if exact else "ชื่อใกล้เคียงกับที่คุณค้นหา"
            movies.append(details)
    seen |= avoid_ids  # หนังที่ผู้ใช้ตอบไปแล้ว ไม่แนะนำซ้ำ (ผลค้นชื่อตรงๆ ยังแสดงตามปกติ)
    for rec, details in zip(recs, found):
        if details and details.get("id") and details["id"] not in seen:
            seen.add(details["id"])
            details["ai_reason"] = rec.get("reason", "")
            movies.append(details)
    return movies, (friendly_error(ai_error) if ai_error else None), ai_info


@st.cache_resource
def _ai_cache():
    """ที่เก็บแคชผล AI (ใช้ร่วมกันทุก session ในโปรเซสเดียวกัน)"""
    return {}


def cached_ai_recommend(query, preferred=None, liked=None, disliked=None):
    """เหมือน ai_recommend แต่แคชคำค้นเดิมไว้ AI_CACHE_TTL วินาที ประหยัดโควตา
    แคชเฉพาะผลที่ AI ตอบสำเร็จ (ไม่แคช error จะได้ลองใหม่ได้)"""
    cache = _ai_cache()
    key = (
        norm_text(query), preferred,
        tuple(sorted(m["id"] for m in (liked or []))), tuple(sorted(m["id"] for m in (disliked or []))),
    )
    now = time.time()
    hit = cache.get(key)
    if hit and now - hit[0] < AI_CACHE_TTL:
        movies, err, info = hit[1]
        return movies, err, {**info, "cached": True}

    movies, err, info = ai_recommend(query, preferred, liked=liked, disliked=disliked)
    if movies and not err and info:
        if len(cache) > 200:  # กันแคชโตไม่จำกัด: ลบอันเก่าสุด
            cache.pop(min(cache, key=lambda k: cache[k][0]))
        cache[key] = (now, (movies, err, info))
    return movies, err, info


# ---------- ประเมินระบบ (โหมดนักพัฒนา เปิดด้วย ?dev=1 หรือ SHOW_EVAL=1) ----------
EVAL_QUERIES = [
    "หนังตลก ดูแล้วอารมณ์ดี ฮาจนปวดท้อง",
    "หนังดราม่าซึ้งๆ ดูแล้วร้องไห้หนัก",
    "หนังผีสยองขวัญ หลอนๆ สร้างบรรยากาศดี",
    "หนังไซไฟ โลกอนาคต พล็อตหักมุม",
    "หนังรักโรแมนติก ซีนสวยๆ",
    "หนังแอคชั่นมันส์ๆ ฉากต่อสู้เยอะ",
    "หนังสืบสวนฆาตกรรม หักมุมตอนจบ",
    "หนังครอบครัว ดูกับเด็กได้",
    "แอนิเมชันที่ผู้ใหญ่ดูแล้วซึ้ง",
    "หนังไทยที่คนไม่ค่อยรู้จักแต่คะแนนดี",
    "หนังสงครามที่สมจริง",
    "หนังเอาชีวิตรอดในป่าหรือกลางทะเล",
    "หนังที่สร้างจากเรื่องจริง",
    "หนังตลกร้ายเสียดสีสังคม",
    "หนังเกาหลีระทึกขวัญ",
    "หนังญี่ปุ่นอบอุ่นหัวใจ",
    "หนังแฟนตาซีผจญภัยโลกเวทมนตร์",
    "หนังอาชญากรรมแก๊งมาเฟีย",
    "หนังสารคดีที่น่าสนใจ",
    "หนังใหม่ที่เพิ่งออกฉาย น่าดู",
    "หนังที่ดูจบแล้วคิดตามไปอีกหลายวัน",
    "หนังกีฬาสร้างแรงบันดาลใจ",
    "หนังซอมบี้หรือสัตว์ประหลาด",
    "หนังแนวติดอยู่ในห้องปิดตาย",
    "หนังเพลงหรือมิวสิคัล",
]


def run_eval(queries, preferred=None, check_repeat=False, on_progress=None):
    """รันคำค้นทดสอบกับ AI ล้วนๆ (ไม่ใช้แคช ไม่ใช้การค้นชื่อตรงๆ) แล้ววัดผลทีละคำค้น"""
    rows = []
    for i, q in enumerate(queries):
        row = {"คำค้น": q}
        try:
            stats = {}
            t0 = time.time()
            movies, _err, info = ai_recommend(q, preferred, include_direct=False, stats=stats)
            row["เวลา (วิ)"] = round(time.time() - t0, 1)
            row["AI เสนอ"] = stats.get("proposed", 0)
            row["หาเจอใน TMDB"] = stats.get("resolved", 0)
            row["หนังไทย"] = sum(1 for m in movies if m.get("language") == "th")
            row["โมเดล"] = f"{info['provider']} · {info['model']}" if info else "-"
            if check_repeat:
                movies2, _e2, _i2 = ai_recommend(q, preferred, include_direct=False)
                a, b = {m["id"] for m in movies}, {m["id"] for m in movies2}
                row["ผลซ้ำกัน (0-1)"] = round(len(a & b) / len(a | b), 2) if (a | b) else None
        except Exception as e:
            row["error"] = friendly_error(e)
        rows.append(row)
        if on_progress:
            on_progress((i + 1) / len(queries))
    return rows


def summarize_eval(rows):
    ok = [r for r in rows if "error" not in r]
    proposed = sum(r["AI เสนอ"] for r in ok)
    resolved = sum(r["หาเจอใน TMDB"] for r in ok)

    def avg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "total": len(rows),
        "errors": len(rows) - len(ok),
        "resolve_rate": (resolved / proposed) if proposed else None,
        "avg_time": avg("เวลา (วิ)"),
        "avg_thai": avg("หนังไทย"),
        "consistency": avg("ผลซ้ำกัน (0-1)"),
    }


def on_platforms(movie, selected, include_rent=False):
    """หนังเรื่องนี้สตรีมได้บนแพลตฟอร์มที่เลือก (อย่างน้อยหนึ่งตัว) ในไทยหรือไม่"""
    names = list(movie.get("providers", []))
    if include_rent:
        names += movie.get("rent_buy", [])
    text = " ".join(names).lower()
    return any(k in text for p in selected for k in PLATFORMS[p])


# ---------------------------------------------------------
# 5. UI Rendering Components (HTML)
# ---------------------------------------------------------
def esc(text):
    return html.escape(str(text), quote=True)


def provider_badges(providers, limit=None):
    items = providers[:limit] if limit else providers
    return "".join(f"<span class='prov-badge'>{esc(p)}</span>" for p in items)


def score_pct(rating):
    return f"คะแนน {int(round(rating * 10))}%"


def fmt_runtime(minutes):
    if not minutes:
        return ""
    h, m = divmod(int(minutes), 60)
    return f"{h} ชม. {m} น." if h else f"{m} น."


def model_badge(info):
    """ป้ายบอกว่าผลลัพธ์นี้มาจากโมเดลไหน (และสลับมาจากเจ้าไหน ถ้ามี)"""
    if not info:
        return ""
    note = ""
    if info.get("skipped"):
        note += (
            f"<span class='model-note'>⚠ สลับจาก {esc(', '.join(info['skipped']))} "
            "(โควตาหมด/ใช้ไม่ได้) อัตโนมัติ</span>"
        )
    if info.get("cached"):
        note += "<span class='model-note ok'>⚡ ผลจากแคช ไม่เสียโควตา</span>"
    chosen = "<span class='model-tag'>เลือกเอง</span>" if info.get("chosen") else ""
    return (
        f"<div class='model-badge'><span class='model-dot'></span>"
        f"<span>{esc(info['provider'])} · <b>{esc(info['model'])}</b></span>{chosen}{note}</div>"
    )


def chain_badge():
    """แสดงลำดับโมเดลที่พร้อมใช้งาน"""
    chain = describe_chain()
    if not chain:
        return ""
    names = " → ".join(f"{p} <b>{esc(m)}</b>" for p, m in chain)
    return f"<div class='model-badge'><span class='model-dot'></span><span>AI ที่พร้อมใช้ (สำรองอัตโนมัติ): {names}</span></div>"


def card_html(m):
    poster = m.get("poster") or PLACEHOLDER_POSTER
    overview = m["overview"]
    if len(overview) > 140:
        overview = overview[:140].rstrip() + "…"
    return (
        f"<a class='card' href='https://www.themoviedb.org/movie/{m['id']}' target='_blank' rel='noopener' "
        f"title='{esc(m['title'])}'>"
        f"<img src='{esc(poster)}' alt='{esc(m['title'])}' loading='lazy'>"
        f"<div class='card-info'>"
        f"<div class='card-title'>{esc(m['title'])}</div>"
        f"<div class='card-meta'><span class='score'>{score_pct(m['rating'])}</span>"
        f"<span>{esc(m['release_date'])}</span></div>"
        f"<div class='card-ov'>{esc(overview)}</div>"
        f"<div>{provider_badges(m.get('providers', []), 2)}</div>"
        f"</div></a>"
    )


def render_row(title, movies):
    if not movies:
        return
    cards = "".join(card_html(m) for m in movies)
    st.markdown(
        f"<div class='row-title'>{esc(title)}</div><div class='row'>{cards}</div>",
        unsafe_allow_html=True,
    )


def render_top10(title, movies):
    if not movies:
        return
    items = "".join(
        f"<div class='top-item'><span class='top-num'>{i}</span>{card_html(m)}</div>"
        for i, m in enumerate(movies[:10], start=1)
    )
    st.markdown(
        f"<div class='row-title'>{esc(title)}</div><div class='row'>{items}</div>",
        unsafe_allow_html=True,
    )


def toggle_pref(kind, m):
    """กด 👍/👎 ซ้ำ = ยกเลิก; กดอีกปุ่ม = เปลี่ยนความเห็น (เก็บใน session ของผู้ใช้คนนั้นเท่านั้น)"""
    other = "dislikes" if kind == "likes" else "likes"
    st.session_state.setdefault(other, {}).pop(m["id"], None)
    bucket = st.session_state.setdefault(kind, {})
    if m["id"] in bucket:
        bucket.pop(m["id"])
    else:
        bucket[m["id"]] = {"id": m["id"], "title": m["title"], "year": m.get("release_date", "")}


def clear_prefs():
    st.session_state.pop("likes", None)
    st.session_state.pop("dislikes", None)


def current_prefs():
    """(หนังที่กดชอบ, หนังที่กดไม่ชอบ) เป็นลิสต์ของ dict {id, title, year}"""
    return (
        list(st.session_state.get("likes", {}).values()),
        list(st.session_state.get("dislikes", {}).values()),
    )


def ai_card_html(m):
    poster = m.get("poster") or PLACEHOLDER_POSTER
    provs = provider_badges(m.get("providers", []))
    rent = m.get("rent_buy", [])
    if rent:
        provs += (
            "<div class='rent-line'>เช่า/ซื้อได้ที่ "
            + "".join(f"<span class='prov-badge rent'>{esc(p)}</span>" for p in rent[:4])
            + "</div>"
        )
    if not provs:
        provs = "<span class='prov-badge'>ไม่มีข้อมูลสตรีมมิ่งในไทย</span>"
    runtime = fmt_runtime(m.get("runtime"))
    trailer = (
        f"<a class='ai-link' href='{esc(m['trailer'])}' target='_blank' rel='noopener'>▶ ดูตัวอย่าง</a>"
        if m.get("trailer") else ""
    )
    return (
        f"<div class='ai-card'>"
        f"<img src='{esc(poster)}' alt='{esc(m['title'])}' loading='lazy'>"
        f"<div class='ai-body'>"
        f"<div class='ai-title'>{esc(m['title'])}</div>"
        f"<div class='ai-meta'><span class='score'>{score_pct(m['rating'])}</span>"
        f"<span>{esc(m['release_date'])}</span><span>{esc(runtime)}</span></div>"
        f"<div class='ai-genres'>{esc(' / '.join(m.get('genres', [])))}</div>"
        f"<div class='ai-reason'>{esc(m.get('ai_reason', ''))}</div>"
        f"<div>{provs}</div>{trailer}"
        f"</div></div>"
    )


def render_ai_grid(movies, cols_per_row=4):
    """การ์ดผลลัพธ์ + ปุ่ม 👍/👎 ใต้การ์ดแต่ละใบ (ต้องใช้วิดเจ็ต Streamlit จึงเรียงด้วย st.columns)"""
    likes = st.session_state.get("likes", {})
    for start in range(0, len(movies), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, m in zip(cols, movies[start:start + cols_per_row]):
            with col:
                st.markdown(ai_card_html(m), unsafe_allow_html=True)
                b_like, b_dis = st.columns(2)
                liked = m["id"] in likes
                # เปลี่ยน key ตามสถานะ เพื่อให้ CSS ทำปุ่ม "ชอบแล้ว" เป็นสีเขียวได้
                b_like.button(
                    "👍 ชอบแล้ว" if liked else "👍 ชอบ",
                    key=f"{'liked' if liked else 'like'}_{m['id']}",
                    on_click=toggle_pref, args=("likes", m),
                )
                b_dis.button(
                    "👎 ไม่ชอบ", key=f"dislike_{m['id']}",
                    on_click=toggle_pref, args=("dislikes", m),
                )


def render_hero(movie):
    bg = movie.get("backdrop") or movie.get("poster") or ""
    runtime = fmt_runtime(movie.get("runtime"))
    genres = " / ".join(movie.get("genres", []))
    trailer_btn = (
        f"<a class='hero-btn primary' href='{esc(movie['trailer'])}' target='_blank' rel='noopener'>▶ ดูตัวอย่าง</a>"
        if movie.get("trailer") else ""
    )
    st.markdown(
        f"<div class='hero' style=\"background-image:url('{esc(bg)}')\">"
        f"<div class='hero-content'>"
        f"<div class='hero-badge'><b>ฮิตที่สุด</b> ในสัปดาห์นี้</div>"
        f"<h1 class='hero-title'>{esc(movie['title'])}</h1>"
        f"<div class='hero-meta'><span class='score'>{score_pct(movie['rating'])}</span>"
        f"<span>{esc(movie['release_date'])}</span><span>{esc(runtime)}</span><span>{esc(genres)}</span></div>"
        f"<div class='hero-overview'>{esc(movie['overview'])}</div>"
        f"<div style='margin-bottom:.8rem'>{provider_badges(movie.get('providers', []))}</div>"
        f"<div class='hero-actions'>{trailer_btn}"
        f"<a class='hero-btn' href='https://www.themoviedb.org/movie/{movie['id']}' "
        f"target='_blank' rel='noopener'>รายละเอียด</a></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 6. Page Layout
# ---------------------------------------------------------
st.markdown(
    "<div class='nav'><span class='nav-logo'>CINEMATCH</span>"
    "<span class='nav-tag'>แนะนำหนังตรงใจด้วย AI</span></div>",
    unsafe_allow_html=True,
)

if not TMDB_API_KEY and not offline.catalog_available():
    st.warning("⚠️ ยังไม่ได้ตั้งค่า TMDB_API_KEY และไม่มีข้อมูลออฟไลน์ จึงไม่สามารถดึงข้อมูลภาพยนตร์ได้")

# Hero (หนังที่กำลังฮิตอันดับต้นที่มีภาพ backdrop)
trending = load_row("trending/movie/week", limit=10)
hero_movie = next((m for m in trending if m.get("backdrop")), None)
if hero_movie:
    render_hero(hero_movie)

# ---------- Search area ----------
# Mood chips — กดทีเดียวค้นหาเลย
CHIPS = [
    ("😂 ตลกฮาๆ", "หนังตลก ดูแล้วอารมณ์ดี ฮาจนปวดท้อง"),
    ("😭 เรียกน้ำตา", "หนังดราม่าซึ้งๆ ดูแล้วร้องไห้หนัก"),
    ("👻 ผีสยอง", "หนังผีสยองขวัญ หลอนๆ สร้างบรรยากาศดี"),
    ("🚀 ไซไฟ", "หนังไซไฟ โลกอนาคต พล็อตหักมุม"),
    ("🎲 สุ่มให้เลย", "แนะนำหนังดีๆ ที่คนไม่ค่อยรู้จักแต่คะแนนสูง สุ่มแนวมาให้เลย"),
]


def set_query(q):
    st.session_state["pending_query"] = q


def clear_results():
    st.session_state.pop("ai_results", None)


with st.container(key="searchpanel"):
    st.markdown(
        "<div class='search-title'>วันนี้อยากดูหนังแบบไหน?</div>"
        f"<div class='search-sub'>เลือกอารมณ์ พิมพ์ หรือพูดบอก AI ได้เลย แล้ว AI จะเลือกให้ {NUM_RECS} เรื่อง</div>",
        unsafe_allow_html=True,
    )

    with st.container(key="chips"):
        chip_cols = st.columns(len(CHIPS))
        for i, (label, q) in enumerate(CHIPS):
            chip_cols[i].button(label, key=f"chip_{i}", on_click=set_query, args=(q,))

    query_to_run = st.session_state.pop("pending_query", None)

    tab_type, tab_voice = st.tabs(["⌨️ พิมพ์", "🎙️ พูด"])

    with tab_type:
        # ใช้ st.form เพื่อให้กด Enter ในช่องพิมพ์แล้วค้นหาได้เลย
        with st.form("type_form"):
            c_in, c_btn = st.columns([6, 1], vertical_alignment="center")
            user_prompt = c_in.text_input(
                label="Search",
                key="type_text",
                placeholder="🔍  เช่น: หนังแอคชั่นไซไฟ พระเอกฉลาด เอาตัวรอดในอวกาศ",
                label_visibility="collapsed",
            )
            type_submitted = c_btn.form_submit_button("ค้นหา")
        if type_submitted and user_prompt.strip():
            query_to_run = user_prompt.strip()

    with tab_voice:
        st.caption("แตะไมค์เพื่อเริ่มพูด แตะอีกครั้งเพื่อหยุด AI จะถอดเสียงให้ แก้ข้อความได้ก่อนค้นหา")
        audio = st.audio_input("อัดเสียง", label_visibility="collapsed", key="voice_input")
        if audio is not None:
            if not (GEMINI_API_KEY or has_groq() or offline.whisper_ready()):
                st.error("⚠️ กรุณาตั้งค่า GEMINI_API_KEY / GROQ_API_KEY หรือดาวน์โหลดโมเดลถอดเสียงในเครื่อง (--whisper small)")
            else:
                audio_bytes = audio.getvalue()
                audio_id = hash(audio_bytes)
                # ถอดเสียงเฉพาะไฟล์เสียงใหม่ ไม่ทำซ้ำเมื่อหน้าเว็บ rerun
                if st.session_state.get("last_audio_id") != audio_id:
                    st.session_state["last_audio_id"] = audio_id
                    with st.spinner("กำลังฟังสิ่งที่คุณพูด..."):
                        try:
                            heard = transcribe_audio(audio_bytes, audio.type or "audio/wav")
                        except Exception as e:
                            heard = ""
                            st.error(f"ถอดเสียงไม่สำเร็จ: {friendly_error(e)}")
                    # ใส่ข้อความที่ถอดได้ลงในช่องแก้ไข (ต้องตั้งค่าก่อนสร้างช่อง text_input)
                    st.session_state["voice_text"] = heard
                    if not heard:
                        st.warning("ไม่ได้ยินเสียงพูดชัดเจน ลองอัดใหม่ หรือพิมพ์ในแท็บ “พิมพ์” ได้เลย")

        # ช่องแก้ไขข้อความที่ถอดเสียงได้ — กด Enter หรือปุ่มเพื่อค้นหา
        if st.session_state.get("last_audio_id") is not None:
            st.markdown("<div class='voice-label'>🎧 AI ได้ยินว่า (แก้ไขได้ถ้าฟังผิด)</div>", unsafe_allow_html=True)
            with st.form("voice_form"):
                c_in, c_btn = st.columns([6, 1], vertical_alignment="center")
                voice_prompt = c_in.text_input(
                    label="ข้อความที่ถอดได้",
                    key="voice_text",
                    placeholder="ข้อความที่ถอดได้จะขึ้นตรงนี้",
                    label_visibility="collapsed",
                )
                voice_submitted = c_btn.form_submit_button("ค้นหา")
            if voice_submitted and voice_prompt.strip():
                query_to_run = voice_prompt.strip()

    # ตั้งค่า AI — พับเก็บไว้ ผู้ใช้ทั่วไปไม่ต้องเห็น
    # เลือกโมเดลเองได้ ถ้าตัวที่เลือกโควตาหมด/ใช้ไม่ได้ จะสลับอัตโนมัติ
    preferred = None
    with st.expander("⚙️ ตั้งค่า AI"):
        st.markdown(chain_badge(), unsafe_allow_html=True)

        _off = st.toggle("โหมดออฟไลน์ (ไม่ใช้อินเทอร์เน็ต)", value=offline.forced(), key="offline_toggle")
        if _off != offline.forced():
            offline.set_forced(_off)
            st.cache_data.clear()      # แคชแถวหนัง/รายชื่อของโหมดเดิม
            _ai_cache().clear()        # แคชผล AI ของโหมดเดิม
            st.session_state.pop("ai_results", None)
            st.rerun()

        _options = model_options()
        if _options:
            _labels = {"auto": "อัตโนมัติ (สลับตามลำดับสำรอง)"}
            for _key, _label, _model in _options:
                _labels[f"{_key}|{_model}"] = f"{_label} · {_model}"
            # กันค่าเก่าค้างใน session ที่ไม่มีในรายการแล้ว (เช่น แก้ .env แล้วโมเดลหาย)
            if st.session_state.get("model_choice") not in _labels:
                st.session_state["model_choice"] = "auto"
            _picked = st.selectbox(
                "โมเดล AI ที่ใช้แนะนำหนัง",
                options=list(_labels),
                format_func=_labels.get,
                key="model_choice",
                help="เลือกโมเดลที่อยากใช้ ถ้าโมเดลนั้นโควตาหมดหรือใช้ไม่ได้ ระบบจะสลับไปตัวสำรองให้เอง",
            )
            if _picked != "auto":
                preferred = tuple(_picked.split("|", 1))

# ---------- โหมดนักพัฒนา: ประเมินระบบ ----------
try:
    DEV_MODE = os.getenv("SHOW_EVAL") == "1" or st.query_params.get("dev") == "1"
except Exception:
    DEV_MODE = os.getenv("SHOW_EVAL") == "1"

if DEV_MODE:
    with st.expander("🧪 ประเมินระบบ (โหมดนักพัฒนา)"):
        st.caption(
            "รันคำค้นทดสอบกับ AI ล้วนๆ (ไม่ใช้แคช) วัด 4 อย่าง: หนังที่ AI เสนอหาเจอจริงใน TMDB กี่ %, "
            "เวลาตอบ, จำนวนหนังไทย, และความคงที่เมื่อถามซ้ำ — ไม่ได้วัดว่า 'ตรงใจ' ซึ่งต้องให้คนตัดสิน "
            "แต่ละคำค้นเรียก AI 1 ครั้ง (2 ครั้งถ้าเปิดทดสอบความคงที่) ระวังโควตา"
        )
        n_q = st.slider("จำนวนคำค้นที่ใช้ทดสอบ", 5, len(EVAL_QUERIES), 10, key="eval_n")
        rep = st.checkbox("ทดสอบความคงที่ด้วย (ถามซ้ำทุกคำค้น ใช้โควตาเป็นสองเท่า)", key="eval_rep")
        if st.button("▶ เริ่มทดสอบ", key="eval_run"):
            bar = st.progress(0.0, text="กำลังทดสอบ...")
            rows = run_eval(
                EVAL_QUERIES[:n_q], preferred, rep,
                on_progress=lambda x: bar.progress(x, text=f"กำลังทดสอบ... {int(x * 100)}%"),
            )
            bar.empty()
            st.session_state["eval_result"] = {"rows": rows, "repeat": rep}

        ev = st.session_state.get("eval_result")
        if ev:
            sm = summarize_eval(ev["rows"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("หาเจอใน TMDB", f"{sm['resolve_rate'] * 100:.0f}%" if sm["resolve_rate"] is not None else "-")
            c2.metric("เวลาตอบเฉลี่ย", f"{sm['avg_time']:.1f} วิ" if sm["avg_time"] is not None else "-")
            c3.metric("หนังไทยเฉลี่ย/คำค้น", f"{sm['avg_thai']:.1f}" if sm["avg_thai"] is not None else "-")
            c4.metric("ผลซ้ำกันเมื่อถามซ้ำ", f"{sm['consistency'] * 100:.0f}%" if sm["consistency"] is not None else "-")
            if sm["errors"]:
                st.warning(f"{sm['errors']} จาก {sm['total']} คำค้นเกิดข้อผิดพลาด (ไม่นับในค่าเฉลี่ย)")
            st.dataframe(ev["rows"], use_container_width=True)

# ---------------------------------------------------------
# 7. Run AI search & show results
# ---------------------------------------------------------
if query_to_run:
    if not has_any_llm():
        st.error("⚠️ กรุณาตั้งค่า API key ของ AI อย่างน้อยหนึ่งเจ้า (GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY) ใน .env")
    else:
        with st.spinner("AI กำลังคัดเลือกหนังให้คุณ..."):
            try:
                _liked, _disliked = current_prefs()
                movies, ai_error, ai_info = cached_ai_recommend(query_to_run, preferred, _liked, _disliked)
                st.session_state["ai_results"] = {
                    "query": query_to_run,
                    "movies": list(movies),
                    "rounds": 0,
                    "ai_error": ai_error,
                    "ai_info": ai_info,
                }
            except Exception as e:
                st.error(f"ระบบ AI ขัดข้องชั่วคราว: {friendly_error(e)}")

results = st.session_state.get("ai_results")
if results:
    st.markdown(
        f"<div class='ai-head'>เลือกให้ตามคำขอ <span>“{esc(results['query'])}”</span></div>"
        f"{model_badge(results.get('ai_info'))}",
        unsafe_allow_html=True,
    )
    st.button("✕ ล้างผลลัพธ์", key="clear_results", on_click=clear_results)
    if results.get("ai_error"):
        st.warning(f"AI แนะนำหนังเพิ่มไม่ได้ในตอนนี้ จึงแสดงเฉพาะหนังที่ชื่อตรง/ใกล้เคียง — {results['ai_error']}")

    # ตัวกรองแพลตฟอร์ม (กรองจากผลที่ได้แล้ว ไม่เรียก AI ซ้ำ)
    selected = st.pills(
        "ดูได้ใน",
        list(PLATFORMS),
        selection_mode="multi",
        key="platform_filter",
    )
    include_rent = st.toggle("รวมที่เช่า/ซื้อได้ด้วย", key="include_rent")
    likes = st.session_state.get("likes", {})
    dislikes = st.session_state.get("dislikes", {})
    if likes or dislikes:
        st.caption(
            f"👍 ชอบ {len(likes)} เรื่อง · 👎 ไม่ชอบ {len(dislikes)} เรื่อง (ซ่อนจากรายการแล้ว) "
            "— AI จะใช้ปรับคำแนะนำรอบถัดไป เก็บไว้เฉพาะในหน้านี้ รีเฟรชแล้วหาย"
        )
        c_retaste, c_clear, _sp = st.columns([2, 1.4, 6])
        c_retaste.button("🔄 แนะนำใหม่ตามรสนิยม", key="retaste_btn", on_click=set_query, args=(results["query"],))
        c_clear.button("ล้างความชอบ", key="clear_prefs", on_click=clear_prefs)
    visible = [m for m in results["movies"] if m["id"] not in dislikes]
    movies = visible
    if selected:
        movies = [m for m in movies if on_platforms(m, selected, include_rent)]
        how = "สตรีมหรือเช่า/ซื้อได้" if include_rent else "สตรีมได้"
        st.caption(
            f"แสดง {len(movies)} จาก {len(visible)} เรื่อง "
            f"ที่{how}ใน {', '.join(selected)} (ประเทศไทย)"
        )

    if movies:
        render_ai_grid(movies)
    elif selected:
        st.info("ไม่มีเรื่องไหนในชุดนี้ที่ดูได้บนแพลตฟอร์มที่เลือก ลองเอาตัวกรองออก เปิด 'รวมที่เช่า/ซื้อ' หรือกดดูเพิ่ม")
    else:
        st.warning("ไม่พบภาพยนตร์ที่ตรงกับคำค้นหา ลองเปลี่ยนคำอธิบายแล้วค้นหาอีกครั้ง")

    # ดูเพิ่ม: ขอ AI อีกชุดโดยห้ามซ้ำกับที่แสดงไปแล้ว (สูงสุด MAX_MORE_ROUNDS ครั้ง)
    rounds = results.get("rounds", 0)
    if results["movies"] and not results.get("ai_error") and rounds < MAX_MORE_ROUNDS:
        if st.button(f"➕ ดูเพิ่มอีก {NUM_RECS} เรื่อง", key="more_btn"):
            with st.spinner("AI กำลังหาเรื่องอื่นให้..."):
                try:
                    _liked, _disliked = current_prefs()
                    more, _err, _info = ai_recommend(
                        results["query"], preferred,
                        exclude_movies=results["movies"], include_direct=False,
                        liked=_liked, disliked=_disliked,
                    )
                    if more:
                        results["movies"] = results["movies"] + more
                        results["rounds"] = rounds + 1
                        st.rerun()
                    else:
                        st.info("AI หาเรื่องใหม่ที่ไม่ซ้ำเพิ่มไม่ได้แล้ว ลองเปลี่ยนคำค้น")
                except Exception as e:
                    st.error(f"ดูเพิ่มไม่สำเร็จ: {friendly_error(e)}")

# ---------------------------------------------------------
# 8. Browse Rows
# ---------------------------------------------------------
render_top10("10 อันดับหนังฮิตในสัปดาห์นี้", trending)
render_row("หนังไทยมาแรง", load_row("discover/movie", (("with_original_language", "th"), ("sort_by", "popularity.desc"), ("vote_count.gte", "10"))))
render_row("หนังผีไทย", load_row("discover/movie", (("with_original_language", "th"), ("with_genres", "27"), ("sort_by", "popularity.desc"), ("vote_count.gte", "5"))))
render_row("หนังคะแนนสูงตลอดกาล", load_row("movie/top_rated"))
render_row("แอคชั่นสุดระห่ำ", load_row("discover/movie", (("with_genres", "28"),)))
render_row("ตลก ดูสนุก ผ่อนคลาย", load_row("discover/movie", (("with_genres", "35"),)))
render_row("แอนิเมชันสำหรับครอบครัว", load_row("discover/movie", (("with_genres", "16"),)))

# Footer
st.markdown(
    "<div class='footer'>พัฒนาด้วย Streamlit · Gemini / Groq / OpenRouter · TMDB API<br>"
    "ข้อมูลสตรีมมิ่งในไทยจาก JustWatch · โปรเจกต์เพื่อการศึกษา</div>",
    unsafe_allow_html=True,
)