"""
Video Analysis Module
Extracts keyframes from video files and runs Azure Vision on each frame.
Also extracts video-level metadata (duration, fps, codec, embedded GPS).
"""

import os
import cv2
import subprocess
import json
from modules.azure_vision import analyse_image
from modules.exif_extractor import extract_exif


def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    result = {
        "duration_sec": None,
        "fps": None,
        "resolution": None,
        "codec": None,
        "creation_time": None,
        "location": None,
        "file_size_mb": None,
        "raw": {}
    }

    if not os.path.exists(video_path):
        return result

    result["file_size_mb"] = round(os.path.getsize(video_path) / (1024 * 1024), 2)

    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            meta = json.loads(out.stdout)
            result["raw"] = meta

            fmt = meta.get("format", {})
            result["duration_sec"] = round(float(fmt.get("duration", 0)), 1)

            tags = fmt.get("tags", {})
            result["creation_time"] = tags.get("creation_time") or tags.get("date")

            # Embedded GPS (common in iPhone videos)
            location_tag = (
                tags.get("location") or
                tags.get("com.apple.quicktime.location.ISO6709") or
                tags.get("location-eng")
            )
            if location_tag:
                result["location"] = _parse_iso6709(location_tag)

            for stream in meta.get("streams", []):
                if stream.get("codec_type") == "video":
                    result["codec"] = stream.get("codec_name")
                    w = stream.get("width")
                    h = stream.get("height")
                    if w and h:
                        result["resolution"] = f"{w}x{h}"
                    r_frame_rate = stream.get("r_frame_rate", "0/1")
                    try:
                        num, den = r_frame_rate.split("/")
                        result["fps"] = round(int(num) / int(den), 2)
                    except Exception:
                        pass
                    break
    except FileNotFoundError:
        result["note"] = "ffprobe not found — install ffmpeg for video metadata."
    except Exception as e:
        result["note"] = f"ffprobe error: {e}"

    return result


def _parse_iso6709(loc_str: str) -> dict:
    """Parse ISO 6709 location string like '+28.6139+077.2090+216/'."""
    try:
        loc_str = loc_str.replace("/", "").strip()
        import re
        parts = re.findall(r'[+-]\d+\.?\d*', loc_str)
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            alt = float(parts[2]) if len(parts) > 2 else None
            return {
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
                "google_maps_url": f"https://www.google.com/maps?q={lat},{lon}"
            }
    except Exception:
        pass
    return {"raw": loc_str}


def extract_keyframes(video_path: str, max_frames: int = 12, method: str = "interval") -> list:
    """
    Extract keyframes from video.
    method: 'interval' (every N seconds) or 'scene' (scene change detection)
    Returns list of saved frame paths.
    """
    if not os.path.exists(video_path):
        return []

    out_dir = f"/tmp/osint_frames_{os.path.basename(video_path).replace('.', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_sec = total_frames / fps

    frame_paths = []

    if method == "interval":
        # Pick frames spread across video duration
        interval = max(1, int(total_frames / max_frames))
        frame_indices = list(range(0, total_frames, interval))[:max_frames]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                timestamp_sec = round(idx / fps, 1)
                out_path = os.path.join(out_dir, f"frame_{idx:06d}_t{timestamp_sec}s.jpg")
                cv2.imwrite(out_path, frame)
                frame_paths.append((out_path, timestamp_sec))

    elif method == "scene":
        # Scene change detection using frame difference
        prev_frame = None
        scene_frames = []
        frame_idx = 0
        threshold = 30.0  # Mean pixel difference threshold

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                score = diff.mean()
                if score > threshold:
                    scene_frames.append((frame_idx, frame))
            else:
                scene_frames.append((frame_idx, frame))  # Always include first frame

            prev_frame = gray
            frame_idx += 1

        # Subsample to max_frames
        step = max(1, len(scene_frames) // max_frames)
        for i, (idx, frame) in enumerate(scene_frames[::step][:max_frames]):
            timestamp_sec = round(idx / fps, 1)
            out_path = os.path.join(out_dir, f"scene_{i:03d}_t{timestamp_sec}s.jpg")
            cv2.imwrite(out_path, frame)
            frame_paths.append((out_path, timestamp_sec))

    cap.release()
    return frame_paths


def analyse_video(video_path: str, max_frames: int = 8) -> dict:
    """
    Full video OSINT analysis:
    1. Extract video metadata (GPS, creation time, codec, resolution)
    2. Extract keyframes
    3. Run Azure Vision on each keyframe
    Returns aggregated findings.
    """
    result = {
        "video_metadata": {},
        "frames_analysed": 0,
        "frame_results": [],
        "aggregated": {
            "all_tags": {},
            "all_objects": {},
            "all_landmarks": [],
            "all_ocr_text": [],
            "all_faces": [],
            "all_brands": []
        }
    }

    # Step 1: Video metadata
    result["video_metadata"] = get_video_metadata(video_path)

    # Step 2: Extract keyframes
    frames = extract_keyframes(video_path, max_frames=max_frames, method="interval")
    result["frames_analysed"] = len(frames)

    if not frames:
        result["error"] = "No frames could be extracted. Check if OpenCV is installed and file is a valid video."
        return result

    # Step 3: Analyse each frame
    for frame_path, timestamp in frames:
        frame_analysis = analyse_image(frame_path)
        frame_analysis["timestamp_sec"] = timestamp
        frame_analysis["frame_path"] = frame_path
        result["frame_results"].append(frame_analysis)

        # Aggregate tags
        for tag in frame_analysis.get("tags", []):
            name = tag["name"]
            if name not in result["aggregated"]["all_tags"]:
                result["aggregated"]["all_tags"][name] = tag["confidence"]
            else:
                result["aggregated"]["all_tags"][name] = max(
                    result["aggregated"]["all_tags"][name], tag["confidence"]
                )

        # Aggregate objects
        for obj in frame_analysis.get("objects", []):
            name = obj["object"]
            result["aggregated"]["all_objects"][name] = result["aggregated"]["all_objects"].get(name, 0) + 1

        # Landmarks
        for lm in frame_analysis.get("landmarks", []):
            if lm not in result["aggregated"]["all_landmarks"]:
                result["aggregated"]["all_landmarks"].append(lm)

        # OCR
        for line in frame_analysis.get("ocr_text", []):
            if line and line not in result["aggregated"]["all_ocr_text"]:
                result["aggregated"]["all_ocr_text"].append(line)

        # Faces
        result["aggregated"]["all_faces"].extend(frame_analysis.get("faces", []))

        # Brands
        for brand in frame_analysis.get("brands", []):
            if brand not in result["aggregated"]["all_brands"]:
                result["aggregated"]["all_brands"].append(brand)

    # Sort tags by confidence
    result["aggregated"]["all_tags"] = dict(
        sorted(result["aggregated"]["all_tags"].items(), key=lambda x: x[1], reverse=True)
    )

    return result
