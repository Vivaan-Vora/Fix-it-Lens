# Fix-it Lens

Point your phone at something broken. Gemma 4 identifies the parts, diagnoses the likely fault,
gives a repair-or-replace verdict with rough costs, flags electrical/battery hazards, and walks you
through the repair step by step.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your Gemini API key (aistudio.google.com)
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. For a phone demo, use your laptop's LAN IP
(http://<ip>:8000) on the same WiFi. The camera button works on mobile browsers; if the phone
needs HTTPS for camera, just use "upload from gallery" or run `ngrok http 8000`.

## Structure

```
fixit-lens/
├── app.py            # FastAPI + Gemma 4 call (image + text -> structured JSON)
├── static/index.html # single-page UI
├── requirements.txt
└── .env.example
```

## Stack
- Gemma 4 (`gemma-4-26b-a4b-it`) via the Gemini API, multimodal image + text
- FastAPI, vanilla JS
