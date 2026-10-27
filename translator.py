import base64
import html as html_lib
import os
import re
import time

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="SamvaadAI | Native Language Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nllb-200-distilled-600M")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# -------------------------------------------------------------
# 1. LANGUAGE CONFIGURATION
# -------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "ace_Arab": "Acehnese (Arabic script)", "ace_Latn": "Acehnese (Latin script)", "acm_Arab": "Mesopotamian Arabic",
    "acq_Arab": "Ta'izzi-Adeni Arabic", "aeb_Arab": "Tunisian Arabic", "afr_Latn": "Afrikaans",
    "ajp_Arab": "South Levantine Arabic", "aka_Latn": "Akan", "als_Latn": "Tosk Albanian", "amh_Ethi": "Amharic",
    "apc_Arab": "North Levantine Arabic", "arb_Arab": "Modern Standard Arabic", "arb_Latn": "Modern Standard Arabic (Latin script)",
    "ars_Arab": "Najdi Arabic", "ary_Arab": "Moroccan Arabic", "arz_Arab": "Egyptian Arabic", "asm_Beng": "Assamese",
    "ast_Latn": "Asturian", "awa_Deva": "Awadhi", "ayr_Latn": "Central Aymara", "azb_Arab": "South Azerbaijani",
    "azj_Latn": "North Azerbaijani", "bak_Cyrl": "Bashkir", "bam_Latn": "Bambara", "ban_Latn": "Balinese",
    "bar_Latn": "Bavarian", "bcl_Latn": "Central Bikol", "bem_Latn": "Bemba", "ben_Beng": "Bengali",
    "bho_Deva": "Bhojpuri", "bjn_Arab": "Banjar (Arabic script)", "bjn_Latn": "Banjar (Latin script)",
    "bod_Tibt": "Standard Tibetan", "bos_Latn": "Bosnian", "bug_Latn": "Buginese", "bul_Cyrl": "Bulgarian",
    "cat_Latn": "Catalan", "ceb_Latn": "Cebuano", "ces_Latn": "Czech", "cjk_Latn": "Chokwe", "ckb_Arab": "Central Kurdish",
    "crh_Latn": "Crimean Tatar", "cym_Latn": "Welsh", "dan_Latn": "Danish", "deu_Latn": "German",
    "dik_Latn": "Southwestern Dinka", "dyu_Latn": "Dyula", "dzo_Tibt": "Dzongkha", "ell_Grek": "Greek",
    "eng_Latn": "English", "epo_Latn": "Esperanto", "est_Latn": "Estonian", "eus_Latn": "Basque", "ewe_Latn": "Ewe",
    "fao_Latn": "Faroese", "fij_Latn": "Fijian", "fin_Latn": "Finnish", "fon_Latn": "Fon", "fra_Latn": "French",
    "fur_Latn": "Friulian", "fuv_Latn": "Nigerian Fulfulde", "gla_Latn": "Scottish Gaelic", "gle_Latn": "Irish",
    "glg_Latn": "Galician", "grn_Latn": "Guarani", "guj_Gujr": "Gujarati", "hat_Latn": "Haitian Creole",
    "hau_Latn": "Hausa", "heb_Hebr": "Hebrew", "hin_Deva": "Hindi", "hne_Deva": "Chhattisgarhi", "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian", "hye_Armn": "Armenian", "ibo_Latn": "Igbo", "ilo_Latn": "Iloko", "ind_Latn": "Indonesian",
    "isl_Latn": "Icelandic", "ita_Latn": "Italian", "jav_Latn": "Javanese", "jpn_Jpan": "Japanese", "kab_Latn": "Kabyle",
    "kac_Latn": "Jingpho", "kam_Latn": "Kamba", "kan_Knda": "Kannada", "kas_Arab": "Kashmiri (Arabic script)",
    "kas_Deva": "Kashmiri (Devanagari script)", "kat_Geor": "Georgian", "kaz_Cyrl": "Kazakh", "kbp_Latn": "Kabiye",
    "kea_Latn": "Kabuverdianu", "khm_Khmr": "Khmer", "kik_Latn": "Kikuyu", "kin_Latn": "Kinyarwanda",
    "kir_Cyrl": "Kyrgyz", "kmb_Latn": "Kimbundu", "kmr_Latn": "Northern Kurdish", "knc_Arab": "Central Kanuri (Arabic script)",
    "knc_Latn": "Central Kanuri (Latin script)", "kon_Latn": "Kongo", "kor_Hang": "Korean", "lao_Laoo": "Lao",
    "lij_Latn": "Ligurian", "lim_Latn": "Limburgish", "lin_Latn": "Lingala", "lit_Latn": "Lithuanian", "lmo_Latn": "Lombard",
    "ltz_Latn": "Luxembourgish", "lub_Latn": "Luba-Katanga", "lug_Latn": "Ganda", "luo_Latn": "Luo", "lus_Latn": "Lushai",
    "lvs_Latn": "Standard Latvian", "mag_Deva": "Magahi", "mai_Deva": "Maithili", "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi", "min_Latn": "Minangkabau", "mkd_Cyrl": "Macedonian", "mlt_Latn": "Maltese",
    "mni_Beng": "Manipuri", "mos_Latn": "Mossi", "mri_Latn": "Maori", "mya_Mymr": "Burmese",
    "nld_Latn": "Dutch", "nno_Latn": "Norwegian Nynorsk", "nob_Latn": "Norwegian Bokmål", "npi_Deva": "Nepali",
    "nso_Latn": "Northern Sotho", "nus_Latn": "Nuer", "nya_Latn": "Nyanja", "oci_Latn": "Occitan", "ory_Orya": "Odia",
    "pag_Latn": "Pangasinan", "pan_Guru": "Eastern Punjabi", "pap_Latn": "Papiamento", "pbt_Arab": "Southern Pashto",
    "pes_Arab": "Western Persian", "plt_Latn": "Plateau Malagasy", "pol_Latn": "Polish", "por_Latn": "Portuguese",
    "prs_Arab": "Dari", "quy_Latn": "Ayacucho Quechua", "ron_Latn": "Romanian", "run_Latn": "Rundi", "rus_Cyrl": "Russian",
    "sag_Latn": "Sango", "san_Deva": "Sanskrit", "sat_Olck": "Santali", "scn_Latn": "Sicilian", "shn_Mymr": "Shan",
    "sin_Sinh": "Sinhala", "slk_Latn": "Slovak", "slv_Latn": "Slovenian",
    "smo_Latn": "Samoan", "sna_Latn": "Shona", "snd_Arab": "Sindhi", "som_Latn": "Somali",
    "sot_Latn": "Southern Sotho", "spa_Latn": "Spanish", "srd_Latn": "Sardinian",
    "srp_Cyrl": "Serbian (Cyrillic script)", "ssw_Latn": "Swati", "sun_Latn": "Sundanese",
    "swe_Latn": "Swedish", "swh_Latn": "Swahili", "szl_Latn": "Silesian", "tam_Taml": "Tamil",
    "tat_Cyrl": "Tatar", "tel_Telu": "Telugu", "tgk_Cyrl": "Tajik", "tgl_Latn": "Tagalog",
    "tha_Thai": "Thai", "tir_Ethi": "Tigrinya", "tpi_Latn": "Tok Pisin", "tsn_Latn": "Tswana",
    "tso_Latn": "Tsonga", "tuk_Latn": "Turkmen", "tum_Latn": "Tumbuka", "tur_Latn": "Turkish",
    "twi_Latn": "Twi", "tzm_Tfng": "Central Atlas Tamazight", "uig_Arab": "Uyghur",
    "ukr_Cyrl": "Ukrainian", "umb_Latn": "Umbundu", "urd_Arab": "Urdu", "uzn_Latn": "Northern Uzbek",
    "vec_Latn": "Venetian", "vie_Latn": "Vietnamese", "war_Latn": "Waray", "wol_Latn": "Wolof",
    "xho_Latn": "Xhosa", "ydd_Hebr": "Eastern Yiddish", "yor_Latn": "Yoruba",
    "zho_Hans": "Simplified Chinese", "zho_Hant": "Traditional Chinese", "zul_Latn": "Zulu",
}
LANG_CODES = sorted(SUPPORTED_LANGUAGES, key=lambda c: SUPPORTED_LANGUAGES[c].lower())


def lang_label(code: str) -> str:
    return f"{SUPPORTED_LANGUAGES[code]}  ({code})"


# Quick-start examples for customer-support scenarios
EXAMPLES = {
    "🎧 Support reply": (
        "Thank you for contacting us. Your refund has been processed and will reach your account within 3 to 5 business days.",
        "eng_Latn", "mar_Deva",
    ),
    "🏥 Healthcare": (
        "Please take this medicine twice a day after meals and visit the clinic again if the fever continues.",
        "eng_Latn", "hin_Deva",
    ),
    "🌾 Agriculture": (
        "Farming requires high-quality seeds.",
        "eng_Latn", "tam_Taml",
    ),
}

# -------------------------------------------------------------
# 2. VISUAL ASSETS (inline SVG, works fully offline)
# -------------------------------------------------------------
def b64(data: str) -> str:
    return base64.b64encode(data.encode("utf-8")).decode("ascii")


DOT_PATTERN_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28'>
<circle cx='2' cy='2' r='1.4' fill='white' fill-opacity='0.16'/></svg>
"""

BODY_PATTERN_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'>
<circle cx='2' cy='2' r='1.2' fill='#6366f1' fill-opacity='0.10'/></svg>
"""

GLOBE_SVG = """
<svg viewBox="0 0 420 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Globe with speech bubbles">
  <defs>
    <radialGradient id="g1" cx="35%" cy="30%" r="80%">
      <stop offset="0%" stop-color="#67e8f9"/><stop offset="55%" stop-color="#6366f1"/><stop offset="100%" stop-color="#312e81"/>
    </radialGradient>
    <linearGradient id="b1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#e0e7ff"/>
    </linearGradient>
  </defs>
  <circle cx="210" cy="215" r="150" fill="url(#g1)"/>
  <g fill="none" stroke="white" stroke-opacity="0.45" stroke-width="1.6">
    <ellipse cx="210" cy="215" rx="150" ry="150"/>
    <ellipse cx="210" cy="215" rx="95" ry="150"/>
    <ellipse cx="210" cy="215" rx="40" ry="150"/>
    <line x1="60" y1="215" x2="360" y2="215"/>
    <path d="M78 140 Q210 175 342 140"/><path d="M78 290 Q210 255 342 290"/>
  </g>
  <g fill="white" fill-opacity="0.9">
    <circle cx="150" cy="170" r="4"/><circle cx="262" cy="150" r="4"/>
    <circle cx="285" cy="260" r="4"/><circle cx="175" cy="285" r="4"/>
  </g>
  <g stroke="white" stroke-opacity="0.7" stroke-width="1.5" stroke-dasharray="4 5" fill="none">
    <path d="M150 170 L262 150 L285 260 L175 285 Z"/>
  </g>
  <g>
    <path d="M28 70 h120 a16 16 0 0 1 16 16 v44 a16 16 0 0 1 -16 16 h-70 l-26 24 v-24 h-24 a16 16 0 0 1 -16 -16 v-44 a16 16 0 0 1 16 -16z" fill="url(#b1)"/>
    <text x="88" y="118" font-size="40" font-weight="700" text-anchor="middle" fill="#4338ca" font-family="Inter,Arial,sans-serif">Hello</text>
  </g>
  <g>
    <path d="M270 290 h112 a16 16 0 0 1 16 16 v44 a16 16 0 0 1 -16 16 h-26 v24 l-26 -24 h-60 a16 16 0 0 1 -16 -16 v-44 a16 16 0 0 1 16 -16z" fill="#22d3ee"/>
    <text x="326" y="340" font-size="34" font-weight="700" text-anchor="middle" fill="#083344" font-family='"Noto Sans Devanagari","Mangal",Arial,sans-serif'>नमस्ते</text>
  </g>
</svg>
"""


def find_custom_hero():
    """Optional: drop hero.jpg / hero.png / hero.webp into ./assets to use your own photo."""
    for ext, mime in (("jpg", "jpeg"), ("jpeg", "jpeg"), ("png", "png"), ("webp", "webp")):
        path = os.path.join(ASSETS_DIR, f"hero.{ext}")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode("ascii")
    return None


def render_html(markup: str) -> None:
    """Render raw HTML; strip leading whitespace so Markdown doesn't treat it as a code block."""
    st.markdown(re.sub(r"^[ \t]+", "", markup, flags=re.M), unsafe_allow_html=True)


# -------------------------------------------------------------
# 3. STYLING
# -------------------------------------------------------------
custom_hero = find_custom_hero()
hero_photo_layer = (
    f"linear-gradient(120deg, rgba(15,23,42,0.92), rgba(49,46,129,0.80) 55%, rgba(8,145,178,0.70)), url('{custom_hero}') center/cover no-repeat,"
    if custom_hero
    else ""
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', 'Noto Sans Devanagari', 'Segoe UI', system-ui, sans-serif; }

.stApp {
  background:
    url("data:image/svg+xml;base64,__BODY_PATTERN__"),
    linear-gradient(180deg, #f5f7ff 0%, #eef3fb 45%, #f8fafc 100%);
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* ---------- HERO ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 28px; padding: 56px 56px 48px 56px; color: #fff;
  display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 24px; align-items: center;
  background:
    __HERO_PHOTO__
    url("data:image/svg+xml;base64,__DOT_PATTERN__"),
    radial-gradient(900px 400px at 90% 0%, rgba(34,211,238,0.35), transparent 60%),
    radial-gradient(700px 400px at 0% 100%, rgba(167,139,250,0.35), transparent 60%),
    linear-gradient(120deg, #0f172a 0%, #312e81 55%, #0e7490 100%);
  box-shadow: 0 25px 60px -20px rgba(49,46,129,0.55);
}
.hero .badge {
  display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 600; letter-spacing: .4px;
  background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.28); backdrop-filter: blur(6px);
}
.hero h1 { font-size: 46px; line-height: 1.1; font-weight: 800; margin: 18px 0 14px 0; letter-spacing: -1px; color: #fff; }
.hero h1 span { background: linear-gradient(90deg, #67e8f9, #c4b5fd); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { font-size: 17px; line-height: 1.65; color: rgba(255,255,255,0.86); max-width: 560px; margin: 0 0 22px 0; }
.hero .art { display: flex; justify-content: center; animation: float 6s ease-in-out infinite; }
.hero .art svg { width: 100%; max-width: 340px; filter: drop-shadow(0 20px 30px rgba(0,0,0,0.35)); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  padding: 6px 14px; border-radius: 999px; font-size: 14px; font-weight: 500; color: #fff;
  background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22);
  animation: float 5s ease-in-out infinite;
}
.chip:nth-child(2n) { animation-delay: .6s; } .chip:nth-child(3n) { animation-delay: 1.2s; } .chip:nth-child(5n) { animation-delay: 1.8s; }
@keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }

/* ---------- STATS ---------- */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: -28px 28px 8px 28px; position: relative; z-index: 2; }
.stat {
  background: #fff; border-radius: 18px; padding: 18px 20px; text-align: center;
  box-shadow: 0 12px 30px -12px rgba(30,27,75,0.25); border: 1px solid #e8ecf8;
}
.stat b { display: block; font-size: 26px; font-weight: 800; background: linear-gradient(90deg,#4f46e5,#0891b2); -webkit-background-clip: text; background-clip: text; color: transparent; }
.stat small { color: #64748b; font-size: 13px; font-weight: 500; }

/* ---------- SECTION TITLES ---------- */
.section-title { font-size: 26px; font-weight: 800; color: #0f172a; margin: 34px 0 4px 0; letter-spacing: -.5px; }
.section-sub { color: #64748b; margin-bottom: 18px; font-size: 15px; }

/* ---------- USE-CASE CARDS ---------- */
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.card {
  background: #fff; border-radius: 20px; padding: 22px; border: 1px solid #e8ecf8;
  box-shadow: 0 8px 24px -14px rgba(30,27,75,0.30); transition: transform .2s ease, box-shadow .2s ease;
}
.card:hover { transform: translateY(-5px); box-shadow: 0 18px 36px -16px rgba(79,70,229,0.45); }
.card .icon {
  width: 52px; height: 52px; border-radius: 16px; display: grid; place-items: center; font-size: 26px; margin-bottom: 14px; color: #fff;
}
.card h4 { margin: 0 0 6px 0; font-size: 17px; font-weight: 700; color: #0f172a; }
.card p { margin: 0; font-size: 14px; color: #64748b; line-height: 1.55; }
.g1 { background: linear-gradient(135deg,#6366f1,#8b5cf6); } .g2 { background: linear-gradient(135deg,#06b6d4,#3b82f6); }
.g3 { background: linear-gradient(135deg,#10b981,#06b6d4); } .g4 { background: linear-gradient(135deg,#f59e0b,#ef4444); }

/* ---------- TRANSLATOR PANEL ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,0.92); border-radius: 24px !important; border: 1px solid #e3e8f7 !important;
  box-shadow: 0 20px 50px -24px rgba(49,46,129,0.35); padding: 6px 10px;
}
.panel-title { font-size: 20px; font-weight: 700; color: #0f172a; margin: 4px 0 0 0; }
.panel-sub { color: #64748b; font-size: 14px; margin: 2px 0 10px 0; }
.stTextArea textarea { border-radius: 14px; font-size: 16px; line-height: 1.6; }
.stSelectbox div[data-baseweb="select"] > div { border-radius: 12px; }
.stButton > button { border-radius: 12px; font-weight: 600; transition: all .15s ease; }
.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #4f46e5, #0891b2); border: none; color: #fff; padding: 0.65rem 1rem; font-size: 16px;
  box-shadow: 0 10px 24px -10px rgba(79,70,229,0.7);
}
.stButton > button[kind="primary"]:hover { transform: translateY(-2px); filter: brightness(1.08); }

/* ---------- RESULT CARD ---------- */
.result {
  margin-top: 14px; border-radius: 20px; padding: 22px 26px; color: #0f172a;
  background: linear-gradient(135deg, #eef2ff 0%, #ecfeff 100%); border: 1px solid #c7d2fe;
  border-left: 6px solid #4f46e5;
}
.result .meta { font-size: 12.5px; font-weight: 700; letter-spacing: .8px; text-transform: uppercase; color: #4f46e5; margin-bottom: 8px; }
.result .text { font-size: 22px; line-height: 1.6; white-space: pre-wrap; font-weight: 500; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.pill { padding: 4px 12px; border-radius: 999px; background: #fff; border: 1px solid #dbe2f5; font-size: 12.5px; color: #475569; font-weight: 500; }

/* ---------- STEPS + FOOTER ---------- */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.step { background: #fff; border-radius: 18px; padding: 20px; border: 1px dashed #c7d2fe; }
.step .n { width: 32px; height: 32px; border-radius: 50%; display: grid; place-items: center; font-weight: 800; color: #fff; background: linear-gradient(135deg,#4f46e5,#0891b2); margin-bottom: 10px; }
.step b { color: #0f172a; } .step p { margin: 4px 0 0 0; color: #64748b; font-size: 14px; }
.footer { text-align: center; color: #94a3b8; font-size: 13px; margin-top: 44px; padding-top: 18px; border-top: 1px solid #e2e8f0; }

@media (max-width: 900px) {
  .hero { grid-template-columns: 1fr; padding: 34px 26px; } .hero .art { display: none; } .hero h1 { font-size: 33px; }
  .stats { grid-template-columns: repeat(2, 1fr); margin: 14px 0; } .cards { grid-template-columns: repeat(2, 1fr); } .steps { grid-template-columns: 1fr; }
}
</style>
"""
CSS = (
    CSS.replace("__BODY_PATTERN__", b64(BODY_PATTERN_SVG))
    .replace("__DOT_PATTERN__", b64(DOT_PATTERN_SVG))
    .replace("__HERO_PHOTO__", hero_photo_layer)
)
st.markdown(CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. MODEL LOADING (cached once per server process)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLLB-200 model from local directory...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    return tokenizer, model


def lang_token_id(code: str) -> int:
    """Look up an NLLB language token id (works on old and new transformers versions)."""
    token_id = tokenizer.convert_tokens_to_ids(code)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Language code '{code}' is not supported by this tokenizer.")
    return token_id


def translate(text: str, src_code: str, tgt_code: str, max_length: int, num_beams: int) -> str:
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    lang_token_id(src_code)  # validate the source code
    forced_bos_token_id = lang_token_id(tgt_code)

    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=max(1, num_beams),
            do_sample=False,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


# -------------------------------------------------------------
# 5. SESSION STATE + CALLBACKS
# -------------------------------------------------------------
st.session_state.setdefault("src_lang", "eng_Latn")
st.session_state.setdefault("tgt_lang", "hin_Deva")
st.session_state.setdefault("input_text", "")


def swap_languages():
    st.session_state.src_lang, st.session_state.tgt_lang = st.session_state.tgt_lang, st.session_state.src_lang


def load_example(name: str):
    text, src, tgt = EXAMPLES[name]
    st.session_state.input_text = text
    st.session_state.src_lang = src
    st.session_state.tgt_lang = tgt


# -------------------------------------------------------------
# 6. SETTINGS (fixed defaults; hidden from the UI)
# -------------------------------------------------------------
max_length = 512
num_beams = 4

# -------------------------------------------------------------
# 7. HERO + STATS
# -------------------------------------------------------------
greetings = ["Hello", "नमस्ते", "नमस्कार", "வணக்கம்", "হ্যালো", "Bonjour", "你好", "こんにちは", "مرحبا", "Hola", "Habari", "Merhaba"]
chips_html = "".join(f'<span class="chip">{g}</span>' for g in greetings)

render_html(
    f"""
<div class="hero">
  <div>
    <span class="badge">✨ SamvaadAI</span>
    <span class="badge">✨ AI-powered · Offline · 200+ languages</span>
    <h1>Speak every customer's <span>native language</span></h1>
    <p>Translate between 200+ languages, including many that mainstream tools ignore. Built for customer support, healthcare, education and regional outreach, with every word processed privately on your own machine.</p>
    <div class="chips">{chips_html}</div>
  </div>
  <div class="art">{GLOBE_SVG}</div>
</div>

<div class="stats">
  <div class="stat"><b>200+</b><small>Languages &amp; dialects</small></div>
  <div class="stat"><b>100%</b><small>Offline &amp; private</small></div>
  <div class="stat"><b>600M</b><small>Parameter NLLB model</small></div>
  <div class="stat"><b>&lt; 2s</b><small>Typical sentence speed*</small></div>
</div>
"""
)

# -------------------------------------------------------------
# 8. TRANSLATOR PANEL
# -------------------------------------------------------------
render_html('<div class="section-title">Try the translator</div><div class="section-sub">Pick languages, paste your text and get an instant translation.</div>')

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Model weight mounting failed: {e}")
    st.info(f"Expected the model folder at: {MODEL_PATH}")
    st.stop()

with st.container(border=True):
    render_html('<div class="panel-title">🔤 Translate text</div><div class="panel-sub">Quick start with an example:</div>')
    ex_cols = st.columns(len(EXAMPLES))
    for col, name in zip(ex_cols, EXAMPLES):
        col.button(name, key=f"ex_{name}", on_click=load_example, args=(name,), use_container_width=True)

    c1, c2, c3 = st.columns([5, 1, 5], vertical_alignment="bottom")
    with c1:
        st.selectbox("From", LANG_CODES, key="src_lang", format_func=lang_label)
    with c2:
        st.button("⇄", on_click=swap_languages, help="Swap languages", use_container_width=True)
    with c3:
        st.selectbox("To", LANG_CODES, key="tgt_lang", format_func=lang_label)

    text = st.text_area(
        "Text to translate",
        key="input_text",
        placeholder="Type or paste the customer's message here...",
        height=170,
    )
    st.caption(f"{len(text)} characters · {len(text.split())} words")

    go = st.button("🚀 Translate", type="primary", use_container_width=True)

    if go:
        src, tgt = st.session_state.src_lang, st.session_state.tgt_lang
        if not text.strip():
            st.warning("Please enter some text to translate.")
        else:
            try:
                with st.spinner("Translating..."):
                    t0 = time.perf_counter()
                    result = translate(text, src, tgt, max_length, num_beams)
                    elapsed = time.perf_counter() - t0
                st.session_state.result = {
                    "original": text, "translated": result, "src": src, "tgt": tgt, "secs": elapsed,
                }
            except Exception as e:
                st.session_state.pop("result", None)
                st.error(f"Inference crash sequence: {e}")

    if "result" in st.session_state:
        r = st.session_state.result
        safe = html_lib.escape(r["translated"])
        render_html(
            f"""
<div class="result">
  <div class="meta">{html_lib.escape(SUPPORTED_LANGUAGES[r['src']])} → {html_lib.escape(SUPPORTED_LANGUAGES[r['tgt']])}</div>
  <div class="text">{safe}</div>
  <div class="pill-row">
    <span class="pill">⏱ {r['secs']:.2f}s</span>
    <span class="pill">🔤 {len(r['translated'])} chars</span>
    <span class="pill">🏷 {r['src']} → {r['tgt']}</span>
  </div>
</div>
"""
        )
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ Download translation", r["translated"], file_name="translation.txt", use_container_width=True)
        with d2:
            with st.popover("📋 Copy text", use_container_width=True):
                st.code(r["translated"], language=None)
        with st.expander("Raw response (same shape as the old API)"):
            st.json({
                "original_text": r["original"],
                "translated_text": r["translated"],
                "src_lang": r["src"],
                "src_display_name": SUPPORTED_LANGUAGES[r["src"]],
                "tgt_lang": r["tgt"],
                "tgt_display_name": SUPPORTED_LANGUAGES[r["tgt"]],
            })

# -------------------------------------------------------------
# 9. USE CASES, HOW IT WORKS, FOOTER
# -------------------------------------------------------------
render_html(
    """
<div class="section-title">Where it makes an impact</div>
<div class="section-sub">Native-language communication builds trust and speeds up resolution.</div>
<div class="cards">
  <div class="card"><div class="icon g1">🎧</div><h4>Customer Support</h4><p>Understand tickets, chats and emails in the customer's own language and reply with confidence.</p></div>
  <div class="card"><div class="icon g2">🏥</div><h4>Healthcare</h4><p>Share prescriptions, instructions and health advice clearly with patients across regions.</p></div>
  <div class="card"><div class="icon g3">🎓</div><h4>Education &amp; Govt.</h4><p>Make learning material, forms and public notices accessible in regional languages.</p></div>
  <div class="card"><div class="icon g4">🛒</div><h4>E-commerce &amp; Outreach</h4><p>Serve multilingual markets and rural communities without extra headcount.</p></div>
</div>

<div class="section-title">How it works</div>
<div class="section-sub">Three simple steps, no internet required.</div>
<div class="steps">
  <div class="step"><div class="n">1</div><b>Choose languages</b><p>Select the source and target from 200+ options, including native scripts.</p></div>
  <div class="step"><div class="n">2</div><b>Paste your text</b><p>Customer message, medical note, notice, anything up to 512 tokens.</p></div>
  <div class="step"><div class="n">3</div><b>Get the translation</b><p>The local NLLB-200 model returns an accurate result in seconds.</p></div>
</div>

<div class="footer">
  Designed and developed by Shivraj Jadhav<br>
</div>
"""
)