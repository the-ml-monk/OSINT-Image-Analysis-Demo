"""
Azure Computer Vision Analysis Module
Handles OCR, object detection, landmark identification,
scene description, and face detection via Azure Vision API.
"""

import os
import requests
from PIL import Image
import io
import base64

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "").rstrip("/")
VISION_KEY = os.getenv("AZURE_VISION_KEY", "")


def _get_headers():
    return {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream"
    }


def _read_image_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()


def _to_jpeg_bytes(file_path: str, max_bytes: int = 4 * 1024 * 1024) -> bytes:
    """
    Convert any image (including HEIC/HEIF) to JPEG bytes safe for Azure Vision.
    Also resizes if over Azure's 4MB limit.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    is_heic = ext in (".heic", ".heif")

    if is_heic:
        # Must convert — Azure does not accept HEIC
        img = Image.open(file_path)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
    else:
        data = _read_image_bytes(file_path)
        # If already within limit and a supported format, return as-is
        if len(data) <= max_bytes and ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif"):
            return data
        img = Image.open(file_path)

    # Resize down if still over 4MB
    if len(data) > max_bytes:
        img = Image.open(io.BytesIO(data)) if is_heic else Image.open(file_path)
        scale = 0.7
        while len(data) > max_bytes and scale > 0.1:
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            resized.convert("RGB").save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            scale -= 0.1

    return data


def analyse_image(file_path: str) -> dict:
    """
    Full Azure Vision analysis: description, tags, objects, landmarks, faces, OCR.
    Returns a unified dict of all findings.
    """
    result = {
        "description": None,
        "confidence": None,
        "tags": [],
        "objects": [],
        "landmarks": [],
        "faces": [],
        "ocr_text": [],
        "ocr_full": None,
        "brands": [],
        "categories": [],
        "adult_flags": {},
        "errors": []
    }

    if not VISION_ENDPOINT or not VISION_KEY:
        result["errors"].append("Azure Vision credentials not set.")
        return result

    image_data = _to_jpeg_bytes(file_path)

    # ---- Analyze v3.2 — description, tags, objects, faces, brands, categories ----
    analyze_url = f"{VISION_ENDPOINT}/vision/v3.2/analyze"
    params = {
        "visualFeatures": "Description,Tags,Objects,Faces,Brands,Categories,Adult",
        "details": "Landmarks",
        "language": "en"
    }
    try:
        resp = requests.post(analyze_url, headers=_get_headers(), params=params, data=image_data, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Description
        captions = data.get("description", {}).get("captions", [])
        if captions:
            result["description"] = captions[0].get("text")
            result["confidence"] = round(captions[0].get("confidence", 0) * 100, 1)

        # Tags
        result["tags"] = [
            {"name": t["name"], "confidence": round(t["confidence"] * 100, 1)}
            for t in data.get("tags", [])
            if t["confidence"] > 0.6
        ]

        # Objects
        result["objects"] = [
            {
                "object": o["object"],
                "confidence": round(o["confidence"] * 100, 1),
                "bbox": o.get("rectangle", {})
            }
            for o in data.get("objects", [])
        ]

        # Faces
        result["faces"] = [
            {
                "age": f.get("age"),
                "gender": f.get("gender"),
                "bbox": f.get("faceRectangle", {})
            }
            for f in data.get("faces", [])
        ]

        # Brands
        result["brands"] = [
            {"brand": b["name"], "confidence": round(b["confidence"] * 100, 1)}
            for b in data.get("brands", [])
        ]

        # Categories (includes landmark details)
        cats = data.get("categories", [])
        result["categories"] = [{"name": c["name"], "score": round(c["score"] * 100, 1)} for c in cats]
        for cat in cats:
            detail = cat.get("detail", {})
            for lm in detail.get("landmarks", []):
                result["landmarks"].append({
                    "name": lm["name"],
                    "confidence": round(lm["confidence"] * 100, 1)
                })

        # Adult flags
        adult = data.get("adult", {})
        result["adult_flags"] = {
            "is_adult": adult.get("isAdultContent", False),
            "is_racy": adult.get("isRacyContent", False),
            "adult_score": round(adult.get("adultScore", 0) * 100, 1),
            "racy_score": round(adult.get("racyScore", 0) * 100, 1)
        }

    except requests.HTTPError as e:
        result["errors"].append(f"Analyze API error: {e.response.status_code} — {e.response.text[:200]}")
    except Exception as e:
        result["errors"].append(f"Analyze failed: {e}")

    # ---- OCR — Read API v3.2 (async) ----
    ocr_result = extract_text(file_path)
    result["ocr_text"] = ocr_result.get("lines", [])
    result["ocr_full"] = ocr_result.get("full_text")
    if ocr_result.get("error"):
        result["errors"].append(ocr_result["error"])

    return result


def extract_text(file_path: str) -> dict:
    """
    Azure Read API (OCR) — extracts printed and handwritten text from image.
    """
    result = {"lines": [], "full_text": None, "error": None}

    if not VISION_ENDPOINT or not VISION_KEY:
        result["error"] = "Azure Vision credentials not set."
        return result

    image_data = _to_jpeg_bytes(file_path)

    # Submit read job
    read_url = f"{VISION_ENDPOINT}/vision/v3.2/read/analyze"
    try:
        resp = requests.post(read_url, headers=_get_headers(), data=image_data, timeout=30)
        resp.raise_for_status()
        operation_url = resp.headers.get("Operation-Location")
        if not operation_url:
            result["error"] = "No operation URL returned from OCR API."
            return result
    except Exception as e:
        result["error"] = f"OCR submit failed: {e}"
        return result

    # Poll for result
    import time
    poll_headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    for _ in range(15):
        time.sleep(1)
        try:
            poll = requests.get(operation_url, headers=poll_headers, timeout=15)
            poll.raise_for_status()
            poll_data = poll.json()
            status = poll_data.get("status")
            if status == "succeeded":
                lines = []
                full_parts = []
                for page in poll_data.get("analyzeResult", {}).get("readResults", []):
                    for line in page.get("lines", []):
                        txt = line.get("text", "")
                        lines.append(txt)
                        full_parts.append(txt)
                result["lines"] = lines
                result["full_text"] = "\n".join(full_parts) if full_parts else None
                return result
            elif status == "failed":
                result["error"] = "OCR job failed."
                return result
        except Exception as e:
            result["error"] = f"OCR poll error: {e}"
            return result

    result["error"] = "OCR timed out."
    return result


# Tags that indicate text-bearing content — triggers enhanced OCR
TEXT_INDICATOR_TAGS = {
    "text", "sign", "banner", "poster", "billboard", "document",
    "letter", "newspaper", "book", "menu", "receipt", "ticket",
    "license plate", "number plate", "nameplate", "badge",
    "whiteboard", "blackboard", "screen", "display", "monitor",
    "handwriting", "calligraphy", "logo", "brand", "label",
    "certificate", "diploma", "card", "envelope", "flyer"
}


def should_enhanced_ocr(vision_result: dict) -> bool:
    """
    Decide if enhanced OCR should kick in based on vision analysis.
    Returns True if the image likely contains meaningful text.
    """
    # Check tags for text indicators
    tags = {t["name"].lower() for t in vision_result.get("tags", [])}
    if tags & TEXT_INDICATOR_TAGS:
        return True

    # If basic OCR already found 2+ lines, worth a deeper pass
    if len(vision_result.get("ocr_text", [])) >= 2:
        return True

    # If description mentions text-related words
    desc = (vision_result.get("description") or "").lower()
    if any(w in desc for w in ["sign", "text", "writing", "banner", "poster", "board"]):
        return True

    return False


def enhanced_ocr(file_path: str) -> dict:
    """
    Enhanced OCR pass — uses Azure Read API with higher resolution
    and returns structured results with word-level bounding boxes.
    Also groups text by spatial proximity to reconstruct blocks.
    """
    result = {
        "lines": [],
        "words": [],
        "text_blocks": [],
        "full_text": None,
        "enhanced": True,
        "error": None
    }

    if not VISION_ENDPOINT or not VISION_KEY:
        result["error"] = "Azure Vision credentials not set."
        return result

    image_data = _to_jpeg_bytes(file_path)

    read_url = f"{VISION_ENDPOINT}/vision/v3.2/read/analyze"
    params = {"readingOrder": "natural"}

    try:
        resp = requests.post(
            read_url, headers=_get_headers(),
            params=params, data=image_data, timeout=30
        )
        resp.raise_for_status()
        operation_url = resp.headers.get("Operation-Location")
        if not operation_url:
            result["error"] = "No operation URL returned."
            return result
    except Exception as e:
        result["error"] = f"Enhanced OCR submit failed: {e}"
        return result

    # Poll for results
    import time
    poll_headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    for _ in range(20):
        time.sleep(1)
        try:
            poll = requests.get(operation_url, headers=poll_headers, timeout=15)
            poll.raise_for_status()
            poll_data = poll.json()
            if poll_data.get("status") == "succeeded":
                all_lines = []
                all_words = []

                for page in poll_data.get("analyzeResult", {}).get("readResults", []):
                    for line in page.get("lines", []):
                        line_text = line.get("text", "")
                        line_bbox = line.get("boundingBox", [])
                        all_lines.append({
                            "text": line_text,
                            "bbox": line_bbox,
                            "confidence": _avg_confidence(line.get("words", []))
                        })
                        for word in line.get("words", []):
                            all_words.append({
                                "text": word.get("text", ""),
                                "bbox": word.get("boundingBox", []),
                                "confidence": word.get("confidence", 0)
                            })

                result["lines"] = all_lines
                result["words"] = all_words
                result["full_text"] = "\n".join(l["text"] for l in all_lines)

                # Group into text blocks by vertical proximity
                result["text_blocks"] = _group_into_blocks(all_lines)

                return result
            elif poll_data.get("status") == "failed":
                result["error"] = "Enhanced OCR failed."
                return result
        except Exception as e:
            result["error"] = f"OCR poll error: {e}"
            return result

    result["error"] = "Enhanced OCR timed out."
    return result


def _avg_confidence(words: list) -> float:
    if not words:
        return 0.0
    confs = [w.get("confidence", 0) for w in words]
    return round(sum(confs) / len(confs), 3)


def _group_into_blocks(lines: list, gap_threshold: int = 40) -> list:
    """Group lines into text blocks based on vertical proximity."""
    if not lines:
        return []

    blocks = []
    current_block = [lines[0]]

    for i in range(1, len(lines)):
        prev_bbox = current_block[-1].get("bbox", [])
        curr_bbox = lines[i].get("bbox", [])

        if len(prev_bbox) >= 6 and len(curr_bbox) >= 6:
            prev_bottom = max(prev_bbox[5], prev_bbox[7]) if len(prev_bbox) >= 8 else prev_bbox[5]
            curr_top = min(curr_bbox[1], curr_bbox[3]) if len(curr_bbox) >= 4 else curr_bbox[1]
            gap = abs(curr_top - prev_bottom)

            if gap > gap_threshold:
                blocks.append({
                    "text": "\n".join(l["text"] for l in current_block),
                    "line_count": len(current_block)
                })
                current_block = [lines[i]]
                continue

        current_block.append(lines[i])

    if current_block:
        blocks.append({
            "text": "\n".join(l["text"] for l in current_block),
            "line_count": len(current_block)
        })

    return blocks


def analyse_image_from_bytes(image_bytes: bytes, filename: str = "upload.jpg") -> dict:
    """Analyse image from in-memory bytes (for Streamlit file uploads)."""
    tmp_path = f"/tmp/osint_tmp_{filename}"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)
    result = analyse_image(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return result
