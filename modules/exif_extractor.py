"""
EXIF Metadata Extraction Module
Extracts GPS, device, timestamp and other metadata from image files.
"""

import os
import io
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exifread
from datetime import datetime

# Register HEIC/HEIF support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False


def dms_to_decimal(dms, ref):
    """Convert degrees/minutes/seconds to decimal degrees."""
    degrees = float(dms[0])
    minutes = float(dms[1]) / 60.0
    seconds = float(dms[2]) / 3600.0
    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
        decimal = -decimal
    return round(decimal, 7)


def extract_exif(file_path: str) -> dict:
    """
    Extract all EXIF metadata from an image file.
    Returns a structured dict with GPS, device, and timestamp info.
    """
    result = {
        "has_exif": False,
        "gps": None,
        "device": {},
        "datetime": None,
        "image_info": {},
        "raw_tags": {},
        "privacy_risk": "LOW"
    }

    if not os.path.exists(file_path):
        result["error"] = "File not found"
        return result

    # --- Basic image info via Pillow ---
    try:
        with Image.open(file_path) as img:
            result["image_info"] = {
                "format": img.format,
                "mode": img.mode,
                "width": img.size[0],
                "height": img.size[1],
                "file_size_kb": round(os.path.getsize(file_path) / 1024, 1)
            }
    except Exception as e:
        result["error"] = f"Cannot open image: {e}"
        return result

    # --- Deep EXIF via exifread (JPG/TIFF) or Pillow EXIF blob (HEIC/HEIF) ---
    is_heic = file_path.lower().endswith((".heic", ".heif"))
    tags = {}

    if is_heic:
        # exifread can't parse HEIC — extract EXIF blob via Pillow and feed it back
        try:
            with Image.open(file_path) as img:
                exif_bytes = img.info.get("exif") or b""
            if exif_bytes:
                # Strip the 'Exif\x00\x00' header if present so exifread can parse it
                if exif_bytes[:6] == b"Exif\x00\x00":
                    exif_bytes = exif_bytes[6:]
                tags = exifread.process_file(io.BytesIO(exif_bytes), details=True)
        except Exception:
            tags = {}
    else:
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
        except Exception:
            tags = {}

    if not tags:
        result["has_exif"] = False
        result["privacy_risk"] = "LOW"
        result["risk_reason"] = "No EXIF metadata found — platform likely stripped it."
        return result

    result["has_exif"] = True

    # --- Device Info ---
    device_fields = {
        "Make": "make",
        "Model": "model",
        "Software": "software",
        "Image Make": "make",
        "Image Model": "model",
    }
    for tag, key in device_fields.items():
        val = tags.get(tag) or tags.get(f"EXIF {tag}") or tags.get(f"Image {tag}")
        if val:
            result["device"][key] = str(val)

    # --- DateTime ---
    for dt_tag in ["EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"]:
        if dt_tag in tags:
            try:
                dt_str = str(tags[dt_tag])
                result["datetime"] = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").isoformat()
            except Exception:
                result["datetime"] = str(tags[dt_tag])
            break

    # --- GPS Extraction ---
    gps_lat = tags.get("GPS GPSLatitude")
    gps_lat_ref = tags.get("GPS GPSLatitudeRef")
    gps_lon = tags.get("GPS GPSLongitude")
    gps_lon_ref = tags.get("GPS GPSLongitudeRef")
    gps_alt = tags.get("GPS GPSAltitude")
    gps_speed = tags.get("GPS GPSSpeed")
    gps_timestamp = tags.get("GPS GPSTimeStamp")
    gps_date = tags.get("GPS GPSDate")

    if gps_lat and gps_lon:
        try:
            lat_values = [v.num / v.den for v in gps_lat.values]
            lon_values = [v.num / v.den for v in gps_lon.values]
            lat = dms_to_decimal(lat_values, str(gps_lat_ref))
            lon = dms_to_decimal(lon_values, str(gps_lon_ref))

            alt = None
            if gps_alt:
                try:
                    alt = round(gps_alt.values[0].num / gps_alt.values[0].den, 2)
                except Exception:
                    pass

            speed = None
            if gps_speed:
                try:
                    speed = round(gps_speed.values[0].num / gps_speed.values[0].den, 2)
                except Exception:
                    pass

            gps_time = None
            if gps_timestamp and gps_date:
                try:
                    ts = gps_timestamp.values
                    h = int(ts[0].num / ts[0].den)
                    m = int(ts[1].num / ts[1].den)
                    s = int(ts[2].num / ts[2].den)
                    gps_time = f"{gps_date} {h:02d}:{m:02d}:{s:02d} UTC"
                except Exception:
                    pass

            result["gps"] = {
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
                "speed_kmh": speed,
                "gps_timestamp": gps_time,
                "google_maps_url": f"https://www.google.com/maps?q={lat},{lon}",
                "gmaps_satellite": f"https://www.google.com/maps/@{lat},{lon},100m/data=!3m1!1e3"
            }
            result["privacy_risk"] = "CRITICAL"
            result["risk_reason"] = "GPS coordinates embedded — exact location recoverable."

        except Exception as e:
            result["gps"] = {"error": f"GPS parse failed: {e}"}

    # --- Assign risk level if GPS not found ---
    if result["privacy_risk"] == "LOW" and result["has_exif"]:
        if result["device"]:
            result["privacy_risk"] = "MEDIUM"
            result["risk_reason"] = "Device model/software exposed — device fingerprinting possible."
        if result["datetime"]:
            if result["privacy_risk"] == "MEDIUM":
                result["risk_reason"] += " Timestamp also present."
            else:
                result["privacy_risk"] = "LOW-MEDIUM"
                result["risk_reason"] = "Timestamp present — activity timing inferable."

    # --- Selected raw tags for display ---
    interesting_tags = [
        "Image Make", "Image Model", "Image Software",
        "EXIF DateTimeOriginal", "EXIF FocalLength",
        "EXIF ISOSpeedRatings", "EXIF Flash",
        "EXIF ExifImageWidth", "EXIF ExifImageLength",
        "GPS GPSLatitude", "GPS GPSLongitude", "GPS GPSAltitude",
        "GPS GPSDate", "GPS GPSTimeStamp"
    ]
    for t in interesting_tags:
        if t in tags:
            result["raw_tags"][t] = str(tags[t])

    return result
