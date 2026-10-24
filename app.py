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
  "tools_needed": ["..."],
  "materials_needed": [{"name": "...", "approx_cost_usd": 0}],
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
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                temperature=0.3,
                response_mime_type="application/json",  # forces structured JSON from Gemma
            ),
        )

        result = extract_json(response.text)

        # Generate a live YouTube video embed URL from Gemma's targeted query
        video_info = result.get("video", {})
        query = video_info.get("youtube_query") or f"how to repair {result.get('item', 'broken object')}"
        encoded_query = urllib.parse.quote_plus(query)

        # Attach embed and search links so the frontend can immediately play the tutorial
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