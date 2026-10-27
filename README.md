# Native Language Translator

A local multilingual translation app built with Streamlit and Meta's NLLB-200 model(hugging_face).

## Features

- Offline translation using a local model folder
- Support for 200+ languages and scripts
- Clean Streamlit UI with language selection and translation output
- Fast API backend for translation requests

## Project files

- `translator.py` — Streamlit web app
- `ai_model.py` — FastAPI backend for translation requests
- `config.toml` — Streamlit configuration
- `nllb-200-distilled-600M/` — local NLLB model files

## Requirements

Install the required Python packages:

```bash
pip install streamlit torch transformers fastapi uvicorn
```

## Run the Streamlit app

```bash
streamlit run translator.py
```

## Run the API server

```bash
python ai_model.py
```

Then access the API at:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/languages`
- `http://127.0.0.1:8000/translate`

## Example API request

```json
{
  "text": "Farming requires high-quality seeds.",
  "src_lang": "eng_Latn",
  "tgt_lang": "hin_Deva"
}
```

## Notes

- The app expects the model folder to exist at `nllb-200-distilled-600M` in the project root.
- Translation quality may vary by language and hardware.
- This project is intended for local, private translation use.
