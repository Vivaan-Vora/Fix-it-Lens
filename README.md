# Fix-it Lens

Every year, millions of household items, from cracked monobloc chairs to appliances with blown fuses, end up in landfills simply because people don’t know what broke, whether it’s worth fixing, or how to start. Fix-it Lens closes the repair confidence gap by putting an AI-powered master technician directly in your pocket.

Powered by Gemma, Fix-it Lens turns any smartphone camera into an intelligent diagnostic scanner:

Visual Fault & Part Recognition: Snap a photo of any broken object. Gemma inspects the damage, isolates fractured components, and diagnoses root causes—differentiating between sudden mechanical stress fractures, UV embrittlement, or electrical wear.
Safety-First Triage: Before anyone picks up a screwdriver, Fix-it Lens flags hidden hazards, such as mains voltage, swollen lithium-ion cells, toxic fumes, or structural failure risks, making the first step a mandatory safety check.
The "Repair or Replace" Economic Verdict: It calculates side-by-side estimated costs for DIY materials versus full replacement value. It gives you a clear financial verdict and tells you exactly how much money you’ll save by fixing it yourself.
Interactive Action Guide & Smart Cart: Follow an interactive step-by-step checklist with progress tracking, complete with exact tool and material specifications linked directly to store searches.
TBD: Gemma-Directed AI Repair Simulation -> Reading static repair steps can be intimidating. Fix-it Lens choreographs a dynamic, time-lapse visual simulation of the exact physical fix, demonstrating how the broken parts clamp, bond, or solder back together in real time.
Fix-it Lens transforms repair from an intimidating chore into an intuitive, visual, and sustainable everyday habit.

## Run

Needs Python 3.10+ (the Gemini SDK that supports Gemma 4 thinking settings does not install on 3.9).

```bash
python3 -m venv .venv && source .venv/bin/activate
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
- Python, HTML, CSS
