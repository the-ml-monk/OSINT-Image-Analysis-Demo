"""
Turn raw GPS coordinates into something an audience understands:
a named place ("Bandra West, Mumbai") and, where possible, the actual
businesses within a few metres of the point.

- place_name(): OpenStreetMap Nominatim reverse geocoding. Free, no key.
- nearby_places(): Serper's Places search (uses the existing SERPER_API_KEY)
  to answer "what is AT these coordinates" — the step that turns a dot on a
  map into "this exact shop / building".
"""

import os
import requests

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_PLACES_URL = "https://google.serper.dev/places"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
# Nominatim's usage policy requires a real User-Agent identifying the app.
_UA = {"User-Agent": "OSINT-Awareness-Demo/1.0 (security lecture)"}


def place_name(lat: float, lon: float) -> dict:
    """Reverse-geocode a point to a readable address. Returns {} on failure."""
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 18, "addressdetails": 1},
            headers=_UA, timeout=15,
        )
        if r.status_code != 200:
            return {}
        d = r.json()
        a = d.get("address", {})
        # Build a short "area, city" label from whatever the address carries.
        area = a.get("suburb") or a.get("neighbourhood") or a.get("quarter") or a.get("village") or a.get("town")
        city = a.get("city") or a.get("state_district") or a.get("county")
        short = ", ".join([p for p in (area, city) if p]) or d.get("display_name", "").split(",")[0]
        return {
            "display_name": d.get("display_name", ""),
            "short": short,
            "name": d.get("name") or "",
            "road": a.get("road", ""),
            "city": city or "",
            "state": a.get("state", ""),
            "country": a.get("country", ""),
            "postcode": a.get("postcode", ""),
        }
    except Exception:
        return {}


def nearby_places(lat: float, lon: float, limit: int = 4) -> list:
    """What businesses sit at this point (Serper Places). Returns [] if no key or none found."""
    if not SERPER_API_KEY:
        return []
    try:
        r = requests.post(
            SERPER_PLACES_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": "places", "ll": f"@{lat},{lon},18z"},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        out = []
        for p in r.json().get("places", [])[:limit]:
            out.append({
                "title": p.get("title", ""),
                "address": p.get("address", ""),
                "category": p.get("category", ""),
                "rating": p.get("rating"),
            })
        return out
    except Exception:
        return []
