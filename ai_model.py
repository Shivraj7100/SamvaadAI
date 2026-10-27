# FastAPI Endpoints

import os
import uvicorn
from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel, Field
from enum import Enum
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
 
app = FastAPI(
 title="NLLB-200 Translation Server",
 description="Translation backend using direct model generation with comprehensive input verification.",
 version="2.2"
)

router = APIRouter(prefix="", tags=["Translation API"])
app.include_router(router)
 
# -------------------------------------------------------------
# 1. CORE LANGUAGE CONFIGURATION
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
 "mni_Beng": "Manipuri", "mos_Latn": "Mossi", "mri_Latn": "Maori", "mnd_Latn": "Mondé", "mya_Mymr": "Burmese",
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
 "zho_Hans": "Simplified Chinese", "zho_Hant": "Traditional Chinese", "zul_Latn": "Zulu"
}
 
# Dynamically construct a validation Enum strictly matching allowed codes
LanguageEnum = Enum("LanguageEnum", {k: k for k in SUPPORTED_LANGUAGES.keys()})
 
# -------------------------------------------------------------
# 2. MODEL ENGINE LIFECYCLE (LOCAL LOADING)
# -------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "nllb-200-distilled-600M")
 
try:
 print(f"Spinning up translation framework from local directory: {MODEL_PATH}...")
 
 tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
 model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
 
 print("AI model and tokenizer loaded successfully.")
except Exception as e:
 print(f"Critical execution error: {str(e)}")
 raise RuntimeError(f"Model weight mounting execution failed: {e}") from e
 
# -------------------------------------------------------------
# 3. TYPING SCHEMAS
# -------------------------------------------------------------
class TranslationRequest(BaseModel):
 text: str = Field(..., min_length=1, description="The textual target to evaluate.")
 src_lang: LanguageEnum = Field(..., description="Valid FLORES-200 starting tag.")
 tgt_lang: LanguageEnum = Field(..., description="Valid FLORES-200 ending tag.")
 
 class Config:
  json_schema_extra = {
    "example": {
    "text": "Farming requires high-quality seeds.",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
    }
    }
 
class TranslationResponse(BaseModel):
 original_text: str
 translated_text: str
 src_lang: str
 src_display_name: str
 tgt_lang: str
 tgt_display_name: str
 
# -------------------------------------------------------------
# 4. ACTIVE ROUTERS
# -------------------------------------------------------------
@router.get("/")
def root_status():
    return {"status": "ok", "service": "NLLB-200 Translation Server"}

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "supported_languages": len(SUPPORTED_LANGUAGES),
    }

@router.get("/languages")
def get_supported_languages():
 """Returns the full dictionary of supported languages and codes."""
 return SUPPORTED_LANGUAGES
 
@router.post("/translate", response_model=TranslationResponse)
async def execute_translation(request: TranslationRequest):
    """Executes translation directly via tokenizer and model generation."""
    try:
      src_str = request.src_lang.value
      tgt_str = request.tgt_lang.value
      if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

      # Set source language on the tokenizer
      tokenizer.src_lang = src_str

      # Tokenize input text
      inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)

      # NLLB expects the target language id from the tokenizer's language map.
      forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_str)
      if forced_bos_token_id is None:
        raise HTTPException(status_code=400, detail=f"Unsupported target language code: {tgt_str}")

      # Generate translation tokens
      translated_tokens = model.generate(
          **inputs,
          forced_bos_token_id=forced_bos_token_id,
          max_length=512,
          num_beams=4,
      )

      # Decode tokens back into text
      translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
      return TranslationResponse(
        original_text=request.text,
        translated_text=translated_text,
        src_lang=src_str,
        src_display_name=SUPPORTED_LANGUAGES[src_str],
        tgt_lang=tgt_str,
        tgt_display_name=SUPPORTED_LANGUAGES[tgt_str]
        )
    except HTTPException:
      raise
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Inference crash sequence: {str(e)}")
 
# -------------------------------------------------------------
# 5. SERVER ENTRYPOINT
# -------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("ai_model:app", host="0.0.0.0", port=8000, reload=False)
