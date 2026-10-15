"""
LLM Analyst Module — Groq-powered intelligence report generation.
Takes all extracted findings (EXIF, Vision, OCR, identity matches)
and produces a structured human-readable intelligence briefing.
"""

import os
import json
import requests

# Gemini if GEMINI_API_KEY is set, else Groq (modules/llm_provider.py). The names are
# kept from the Groq-only version so the calls below read as before.
from modules.llm_provider import LLM_KEY as GROQ_API_KEY, LLM_URL as GROQ_URL, LLM_MODEL, LLM_EXTRA
GROQ_MODEL = LLM_MODEL


def generate_report(
    exif_data: dict,
    vision_data: dict,
    identity_matches: list = None,
    filename: str = "uploaded_image",
    risk_level: str = None
) -> str:
    """
    Generate a structured intelligence report from all extracted findings.
    Returns formatted markdown text.
    """
    if not GROQ_API_KEY:
        return "_LLM not configured — set GEMINI_API_KEY (or GROQ_API_KEY) in .env_"

    # Build a condensed facts sheet for the LLM
    facts = _build_facts(exif_data, vision_data, identity_matches, filename)

    risk_line = f"The data in this photo has already been scored {risk_level}. Use EXACTLY this level — do not lower it." if risk_level else ""
    system_prompt = f"""You are a privacy-exposure analyst for a public security-awareness session.
Your job is to show an ordinary person HOW MUCH a single photo they posted gives away about them —
their location, routine, home, family, devices and interests. You are NOT assessing a military/combat threat;
even a happy family photo can be HIGH risk because of what it exposes.

Rules:
- MAX 110 words. Bullet points, not paragraphs. Bold the most exposing findings.
- Structure: **What** (scene) · **Who** (people — count and any public figure named) · **Where** (the NAMED place if given, else GPS) · **When** (use the LOCAL capture time exactly as given — never relabel it UTC) · **Exposed** (2-3 concrete things an attacker/stalker learns) · **Risk** (level + one-line reason).
{risk_line}
- Frame "Exposed" as what a STRANGER now knows: e.g. "exact building and time they were there", "the phone model and iOS version to target", "a child's school from a uniform".
- No disclaimers, no filler."""

    user_prompt = f"""Brief this image for a privacy-awareness audience — "{filename}":

{facts}"""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                **LLM_EXTRA,
                "max_tokens": 400
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        return f"_LLM error: {e.response.status_code} — {e.response.text[:200]}_"
    except Exception as e:
        return f"_LLM error: {e}_"


def generate_identity_query(vision_data: dict, exif_data: dict) -> list:
    """
    Use LLM to generate smart search queries to identify people in the image.
    Returns a list of search query strings.
    """
    if not GROQ_API_KEY:
        return []

    description = vision_data.get("description", "") or ""
    tags = [t["name"] for t in vision_data.get("tags", [])]
    ocr_lines = vision_data.get("ocr_text", [])
    face_count = len(vision_data.get("faces", []))
    landmarks = [l["name"] for l in vision_data.get("landmarks", [])]
    brands = [b["brand"] for b in vision_data.get("brands", [])]

    gps = exif_data.get("gps", {}) or {}
    lat = gps.get("latitude")
    lon = gps.get("longitude")
    dt = exif_data.get("datetime", "")

    prompt = f"""Based on these image analysis findings, generate 2-4 Google search queries
that could help identify the people or event in this image.

Scene: {description}
Tags: {', '.join(tags[:15])}
Faces detected: {face_count}
OCR text found: {'; '.join(ocr_lines[:10]) if ocr_lines else 'None'}
Landmarks: {', '.join(landmarks) if landmarks else 'None'}
Brands visible: {', '.join(brands) if brands else 'None'}
GPS: {"%.4f, %.4f" % (lat, lon) if lat else "Not available"}
Date: {dt or "Unknown"}

Return ONLY a JSON array of search query strings. No explanation.
Example: ["query 1", "query 2"]

Focus queries on:
- Any event names / hashtags from OCR text
- Landmark + date combinations
- Any organization names or banners visible
- Prominent figure identification if formal attire / official setting detected"""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                **LLM_EXTRA,
                "max_tokens": 300
            },
            timeout=45
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Extract JSON array from response
        if "[" in text:
            json_str = text[text.index("["):text.rindex("]") + 1]
            return json.loads(json_str)
    except Exception:
        pass

    # Fallback: construct basic queries from OCR and scene
    queries = []
    if ocr_lines:
        queries.append(" ".join(ocr_lines[:3]))
    if description:
        queries.append(f"{description} event")
    return queries[:4]


def _build_facts(exif_data, vision_data, identity_matches, filename) -> str:
    """Build a plain-text facts sheet for the LLM."""
    lines = [f"## Extracted Facts — {filename}\n"]

    # EXIF
    lines.append("### Metadata (EXIF)")
    if exif_data.get("has_exif"):
        device = exif_data.get("device", {})
        if device:
            lines.append(f"- Device: {device.get('make', '?')} {device.get('model', '?')}")
            if device.get("software"):
                lines.append(f"- Software: {device['software']}")
        if exif_data.get("datetime"):
            lines.append(f"- Capture time: {exif_data['datetime']}")
        if exif_data.get("datetime_local"):
            lines.append(f"- Capture time (LOCAL): {exif_data['datetime_local']}")
        gps = exif_data.get("gps")
        if gps and "latitude" in gps:
            if gps.get("place"):
                lines.append(f"- NAMED LOCATION: {gps['place']}")
            lines.append(f"- GPS: {gps['latitude']}, {gps['longitude']}")
            if gps.get("altitude_m"):
                lines.append(f"- Altitude: {gps['altitude_m']}m")
            if gps.get("speed_kmh") and gps["speed_kmh"] > 0:
                lines.append(f"- Speed: {gps['speed_kmh']} km/h")
        else:
            lines.append("- GPS: Not embedded")
    else:
        lines.append("- No EXIF metadata found (likely stripped by platform)")

    # Vision
    lines.append("\n### Computer Vision Analysis")
    if vision_data.get("description"):
        lines.append(f"- Scene: {vision_data['description']} ({vision_data.get('confidence', '?')}% confidence)")

    tags = vision_data.get("tags", [])
    if tags:
        lines.append(f"- Tags: {', '.join(t['name'] for t in tags[:12])}")

    objects = vision_data.get("objects", [])
    if objects:
        lines.append(f"- Objects: {', '.join(o['object'] for o in objects)}")

    faces = vision_data.get("faces", [])
    if faces:
        lines.append(f"- Faces detected: {len(faces)} identifiable face(s) — each one reverse-searchable")

    landmarks = vision_data.get("landmarks", [])
    if landmarks:
        lines.append(f"- Landmarks: {', '.join(l['name'] + ' (' + str(l['confidence']) + '%)' for l in landmarks)}")

    brands = vision_data.get("brands", [])
    if brands:
        lines.append(f"- Brands: {', '.join(b['brand'] for b in brands)}")

    ocr = vision_data.get("ocr_text", [])
    if ocr:
        lines.append(f"- OCR text extracted: {' | '.join(ocr[:15])}")

    # Identity matches
    if identity_matches:
        lines.append("\n### Identity / News Matches")
        for match in identity_matches[:5]:
            lines.append(f"- [{match.get('title', 'Link')}]({match.get('link', '#')})")
            if match.get("snippet"):
                lines.append(f"  {match['snippet'][:150]}")

    return "\n".join(lines)
