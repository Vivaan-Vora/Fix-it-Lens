import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")  # or gemma-4-31b-it
client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env

app = FastAPI(title="Fix-it Lens")

SYSTEM_PROMPT = """You are Fix-it Lens, an expert repair technician.
The user sends a photo of a broken item plus an optional description.
Respond with ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{
  "item": "what the object is",
  "parts": ["visible/relevant parts you can identify"],
  "diagnosis": {
    "likely_fault": "most probable cause",
    "confidence": "low | medium | high",
    "other_possibilities": ["..."]
  },
  "steps": ["numbered-step text, one action per string, no leading numbers"],
  "tools_needed": [{"name": "...", "approx_cost_usd": 0, "search_term": "..."}],
  "materials_needed": [{"name": "...", "approx_cost_usd": 0, "search_term": "..."}],
  "resources": {
    "video_query": "YouTube search for a tutorial on this exact repair",
    "guide_query": "search for a written repair guide (e.g. iFixit)"
  },
  "difficulty": "easy | medium | hard",
  "time_estimate": "e.g. 20 minutes",
  "verdict": {
    "decision": "repair | replace | ask a professional",
    "repair_cost_usd": 0,
    "replace_cost_usd": 0,
    "reasoning": "one or two sentences"
  },
  "safety": {
    "level": "none | caution | danger",
    "warnings": ["specific warnings, e.g. unplug first, do not open swollen batteries"]
  },
  "questions": ["optional clarifying questions if the photo is ambiguous"]
}

Rules:
- Costs are rough US-dollar estimates; say so in reasoning if unsure.
- If the fix involves mains electricity, capacitors, lithium batteries, gas, or
  heating elements, safety.level must be "caution" or "danger" and the FIRST step
  must be a safety step (unplug / remove battery / etc.).
- If you cannot identify the item or the fault, say so in diagnosis and fill "questions".
- "search_term" is what to type into a store search to buy the right item: be specific
  (size, type, model, e.g. "700c x 25mm presta inner tube", "#2 phillips screwdriver").
- Never write URLs; the app builds links from your queries.
- Be concise. Max 10 steps.
"""


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


@app.post("/api/diagnose")
async def diagnose(image: UploadFile = File(...), description: str = Form("")):
    try:
        data = await image.read()
        mime = image.content_type or "image/jpeg"
        prompt = f"User description: {description or '(none given)'}"

        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                # "minimal" = thinking off (faster demo). Use "high" for hard cases.
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                temperature=0.3,
            ),
        )
        return extract_json(response.text)
    except Exception as e:  # keep the demo alive; show the error
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
