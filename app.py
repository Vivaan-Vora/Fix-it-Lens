import json
import os
import re
import urllib.parse

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")  # or gemma-4-31b-it
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")  # optional: for AI video generation
client = genai.Client()  # reads GEMINI_API_KEY from environment

app = FastAPI(title="Fix-it Lens")

SYSTEM_PROMPT = """You are Fix-it Lens, an expert repair technician and visual director.
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
  "questions": ["optional clarifying questions if the photo is ambiguous"],
  "video": {
    "simulation_prompt": "Cinematic visual prompt for an Image-to-Video AI. Describe the exact physical repair actions on this item in smooth time-lapse (e.g., hands applying epoxy, clamping cracked joint, seam curing into clean seamless finish).",
    "youtube_query": "Concise, targeted DIY search query to find this exact repair video"
  }
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
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise

    # 1. If Gemma wrapped the entire response in a list [ { ... } ], unpack the first item
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        data = data[0]

    return data


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
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

        result = extract_json(response.text)

        # 2. Defensive check: make sure result is a dict
        if not isinstance(result, dict):
            return JSONResponse({"error": "Gemma did not return a valid JSON object."}, status_code=500)

        # 3. Defensive check: ensure 'video' is a dict, not a list of strings
        video_data = result.get("video")
        if not isinstance(video_data, dict):
            fallback_query = (
                str(video_data[0])
                if isinstance(video_data, list) and len(video_data) > 0
                else f"how to repair {result.get('item', 'broken object')}"
            )
            result["video"] = {
                "simulation_prompt": fallback_query,
                "youtube_query": fallback_query,
            }

        query = result["video"].get("youtube_query") or f"how to repair {result.get('item', 'broken item')}"
        encoded_query = urllib.parse.quote_plus(str(query))

        result["video"]["youtube_embed_url"] = f"https://www.youtube.com/embed?listType=search&list={encoded_query}"
        result["video"]["youtube_search_url"] = f"https://www.youtube.com/results?search_query={encoded_query}"

        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/generate-video")
async def generate_video(prompt: str = Form(...), image: UploadFile = File(None)):
    """
    Optional endpoint: Call this if you want to generate an AI Image-to-Video
    simulation of the repair using an external model (e.g. Replicate MiniMax/Kling).
    """
    try:
        if not REPLICATE_API_TOKEN:
            return JSONResponse({
                "status": "skipped",
                "message": "REPLICATE_API_TOKEN not configured. Use the youtube_embed_url for live video."
            })

        import replicate

        # Example using MiniMax Image-to-Video on Replicate
        input_data = {"prompt": prompt}
        if image:
            image_bytes = await image.read()
            input_data["first_frame_image"] = image_bytes

        output = replicate.run(
            "minimax/video-01",
            input=input_data
        )
        return {"video_url": output}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")