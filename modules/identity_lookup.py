"""
Identity Lookup Module — Uses vision LLM (Groq) to identify prominent figures
directly from the image, then searches Google (Serper) for corroboration.
"""

import os
import json
import base64
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_IMAGES_URL = "https://google.serper.dev/images"


def _image_to_base64(file_path: str, max_dim: int = 1280) -> str:
    """Read image, resize for LLM, return base64 JPEG string."""
    from PIL import Image
    import io
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    img = Image.open(file_path).convert("RGB")
    # Resize if too large — LLMs don't need 4000px images
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _detect_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
             ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/jpeg",
             ".heif": "image/jpeg", ".bmp": "image/bmp", ".tiff": "image/tiff"}
    return types.get(ext, "image/jpeg")


def identify_people_via_vision(file_path: str, vision_context: dict = None) -> dict:
    """
    Send the actual image to a vision LLM to identify prominent figures.
    Returns structured identification results.
    """
    result = {
        "people_identified": [],
        "event_identified": None,
        "venue_identified": None,
        "raw_analysis": "",
        "confidence_note": "",
        "error": None
    }

    if not GROQ_API_KEY:
        result["error"] = "GROQ_API_KEY not set."
        return result

    img_b64 = _image_to_base64(file_path)
    media_type = _detect_media_type(file_path)

    # Build context from prior vision analysis
    context_lines = []
    if vision_context:
        desc = vision_context.get("description", "")
        if desc:
            context_lines.append(f"Scene description: {desc}")
        ocr = vision_context.get("ocr_text", [])
        if ocr:
            context_lines.append(f"Text found in image: {'; '.join(ocr[:10])}")
        faces = vision_context.get("faces", [])
        if faces:
            context_lines.append(f"Faces detected: {len(faces)}")
        landmarks = vision_context.get("landmarks", [])
        if landmarks:
            context_lines.append(f"Landmarks: {', '.join(l['name'] for l in landmarks)}")

    context_str = "\n".join(context_lines) if context_lines else "No prior analysis available."

    prompt = f"""You are an expert OSINT image analyst. Study this image and extract every identifying detail.

FOCUS ON CONTEXTUAL CLUES — read everything visible:

1. **Name tags / badges / lanyards**: Read ANY text on ID cards, name badges, lanyards, chest tags. Zoom in on each person's badge area.
2. **Uniforms & insignia**: Military rank insignia, unit patches, shoulder badges, medals, ribbons — identify force, rank, unit.
3. **Banners / backdrops / standees**: Read ALL text on event banners, podium signs, standees, roll-up displays.
4. **Flags / logos / emblems**: National flags, organizational logos, military crests.
5. **Event type**: Conference, military ceremony, diplomatic meeting, social party, press conference, etc.
6. **Venue clues**: Hotel branding, venue name, architecture style, country indicators.

Prior analysis:
{context_str}

IMPORTANT: Do NOT guess identities from faces. Only name someone if you can read their name on a badge/tag/banner or are extremely certain from other context. Focus on what you can READ, not recognize.

Return as JSON:
{{
  "people": [
    {{"description": "brief description", "name_tag_text": "text from badge or null", "uniform_details": "rank/force/insignia or null", "role_inference": "inferred role", "confidence": "high/medium/low"}}
  ],
  "event": {{"type": "...", "name": "from banners if visible", "date": "if visible", "confidence": "high/medium/low"}},
  "venue": {{"type": "...", "name": "if visible", "country_indicators": "flags/language clues", "confidence": "high/medium/low"}},
  "text_found": ["every readable text fragment from banners, signs, badges"],
  "flags_and_emblems": ["any flags or emblems visible"],
  "key_observations": "2-3 sentence summary"
}}"""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1000
            },
            timeout=45
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        result["raw_analysis"] = text

        # Parse JSON from response
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            parsed = json.loads(json_str)

            result["people_identified"] = parsed.get("people", [])
            result["event_identified"] = parsed.get("event")
            result["venue_identified"] = parsed.get("venue")
            result["text_found"] = parsed.get("text_found", [])
            result["flags_and_emblems"] = parsed.get("flags_and_emblems", [])
            result["additional_observations"] = parsed.get("key_observations", "") or parsed.get("additional_observations", "")

    except json.JSONDecodeError:
        # LLM returned text but not valid JSON — still useful
        result["confidence_note"] = "Response was not structured JSON — showing raw analysis."
    except requests.HTTPError as e:
        result["error"] = f"Vision LLM error: {e.response.status_code} — {e.response.text[:200]}"
    except Exception as e:
        result["error"] = f"Vision LLM error: {e}"

    return result


def corroborate_identity(name: str, role: str = "", event_context: str = "") -> dict:
    """
    After vision LLM identifies someone, search Google for corroboration.
    Returns news articles, images, and knowledge graph info.
    """
    result = {"news": [], "images": [], "knowledge_graph": None, "error": None}

    if not SERPER_API_KEY:
        result["error"] = "SERPER_API_KEY not set."
        return result

    # Build search query
    query = name
    if role:
        query += f" {role}"
    if event_context:
        query += f" {event_context}"

    try:
        # Web search
        resp = requests.post(
            SERPER_SEARCH_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("organic", []):
            result["news"].append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", "")
            })

        for item in data.get("news", []):
            result["news"].append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "date": item.get("date", ""),
                "is_news": True
            })

        kg = data.get("knowledgeGraph")
        if kg:
            result["knowledge_graph"] = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", ""),
                "image_url": kg.get("imageUrl", ""),
                "attributes": kg.get("attributes", {})
            }

        # Image search
        img_resp = requests.post(
            SERPER_IMAGES_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": f"{name} photo", "num": 4},
            timeout=15
        )
        img_resp.raise_for_status()
        for item in img_resp.json().get("images", []):
            result["images"].append({
                "title": item.get("title", ""),
                "imageUrl": item.get("imageUrl", ""),
                "link": item.get("link", ""),
                "source": item.get("source", "")
            })

    except Exception as e:
        result["error"] = str(e)

    return result


def chat_about_image(file_path: str, user_question: str, conversation_history: list = None,
                     analysis_context: str = "") -> str:
    """
    Chatbot: Answer questions about the uploaded image.
    Supports multi-turn conversation with history.
    """
    if not GROQ_API_KEY:
        return "LLM not configured — set GROQ_API_KEY in .env"

    img_b64 = _image_to_base64(file_path)
    media_type = _detect_media_type(file_path)

    system_msg = f"""You are an expert image analyst assisting with image intelligence analysis.
You have been given an image to analyse. Answer the user's questions about it thoroughly.

Prior automated analysis of this image:
{analysis_context if analysis_context else "No prior analysis available."}

Be specific, factual, and helpful. If you're unsure about something, say so.
If asked to identify people, only name those you're genuinely confident about."""

    messages = [{"role": "system", "content": system_msg}]

    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)

    # Current question with image
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_question},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{img_b64}"}
            }
        ]
    })

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_VISION_MODEL,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 800
            },
            timeout=45
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        return f"Error: {e.response.status_code} — {e.response.text[:200]}"
    except Exception as e:
        return f"Error: {e}"
