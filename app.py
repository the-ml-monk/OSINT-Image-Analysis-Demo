"""
Image Analysis Demo — Streamlit Dashboard
AI-powered image/video intelligence with LLM reporting,
identity lookup, enhanced OCR, and metadata extraction.
"""

import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Make modules importable
sys.path.insert(0, os.path.dirname(__file__))

from modules.exif_extractor import extract_exif
from modules.azure_vision import (
    analyse_image_from_bytes, analyse_image,
    should_enhanced_ocr, enhanced_ocr
)
from modules.video_analyser import analyse_video, get_video_metadata
from modules.llm_analyst import generate_report
from modules.llm_provider import PROVIDER, LLM_KEY, check_llm
from modules.geo import place_name
from modules.identity_lookup import identify_people_via_vision, corroborate_identity, chat_about_image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Analysis Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling — compact layout ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Aggressive compaction — fit more on screen */
    html { font-size: 13px !important; }
    .block-container { padding: 1.2rem 1rem 0 1rem !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { width: 200px !important; }
    section[data-testid="stSidebar"] .block-container { padding: 0.5rem !important; }
    h1, .main h1, .main .stMarkdown h1 { font-size: 1.4rem !important; margin: 0.4rem 0 0.4rem 0 !important; line-height: 1.3 !important; }
    h2, h3 { font-size: 1rem !important; margin: 0.4rem 0 0.2rem 0 !important; }
    p, li, .stMarkdown, .stMarkdown p { font-size: 0.8rem !important; margin-bottom: 0.2rem !important; }
    .stMetric { padding: 0.2rem 0 !important; }
    .stMetric label { font-size: 0.7rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1rem !important; }
    div[data-testid="stExpander"] { margin: 0.15rem 0 !important; }
    div[data-testid="stExpander"] summary { font-size: 0.8rem !important; padding: 0.3rem 0.5rem !important; }
    hr, .stDivider { margin: 0.25rem 0 !important; }
    .stCaption { font-size: 0.7rem !important; }
    pre, code { font-size: 0.72rem !important; padding: 3px 6px !important; line-height: 1.3 !important; }
    .stChatMessage { padding: 0.3rem 0.5rem !important; }
    .stChatInput { padding: 0.2rem !important; }
    div[data-testid="stFileUploader"] { padding: 0.3rem !important; }
    div[data-testid="stFileUploader"] section { padding: 0.5rem !important; min-height: 50px !important; }
    div[data-testid="stFileUploader"] small { font-size: 0.7rem !important; }
    button { font-size: 0.78rem !important; padding: 0.25rem 0.6rem !important; }
    [data-testid="column"] { padding: 0 0.25rem !important; }
    [data-testid="stImage"] img { max-height: 400px !important; width: 100% !important; object-fit: contain !important; }
    .stSpinner { font-size: 0.75rem !important; }

    /* Risk badges */
    .risk-critical { background:#ff4b4b; color:white; padding:3px 10px; border-radius:4px; font-weight:bold; font-size:0.8rem; }
    .risk-high     { background:#ff8c00; color:white; padding:3px 10px; border-radius:4px; font-weight:bold; font-size:0.8rem; }
    .risk-medium   { background:#ffd700; color:#333;  padding:3px 10px; border-radius:4px; font-weight:bold; font-size:0.8rem; }
    .risk-low      { background:#28a745; color:white; padding:3px 10px; border-radius:4px; font-weight:bold; font-size:0.8rem; }

    /* Cards */
    .identity-card { border-left:3px solid #ff8c00; padding:6px 10px; margin:4px 0; background:#fff8f0; border-radius:3px; font-size:0.85rem; }
    .ocr-enhanced  { border:1px solid #4a90d9; border-radius:6px; padding:8px; background:#f0f6ff; }
</style>
""", unsafe_allow_html=True)


# ── Access ────────────────────────────────────────────────────────────────────
# Online, a password keeps strangers from using the Vision and Gemini keys. On a
# laptop APP_PASSWORD is not set and the app opens straight away.
_APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if _APP_PASSWORD and not st.session_state.get("authed"):
    import hmac
    st.markdown("## Image Analysis Demo")
    st.caption("Security awareness demonstration — enter the password to continue.")
    _entered = st.text_input("Password", type="password")
    if _entered:
        if hmac.compare_digest(_entered, _APP_PASSWORD):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────
RISK_COLOURS = {
    "CRITICAL": "risk-critical",
    "HIGH":     "risk-high",
    "MEDIUM":   "risk-medium",
    "LOW-MEDIUM":"risk-medium",
    "LOW":      "risk-low"
}

def risk_badge(level: str) -> str:
    cls = RISK_COLOURS.get(level.upper(), "risk-low")
    return f'<span class="{cls}">{level}</span>'


def save_upload(uploaded_file) -> str:
    """Save uploaded file to a temp path and return path."""
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


# ── Cached analysis ───────────────────────────────────────────────────────────
# In Streamlit the whole script re-runs on EVERY interaction (including a chat
# message). Without caching, each chat question silently re-ran the entire Azure
# Vision + Gemini + geocode pipeline before answering — the ~30s freeze. These
# wrappers are keyed by the image bytes, so the heavy work runs once per image
# and a chat message only runs the chat call. Args prefixed "_" are passed but
# not used in the cache key.
def _bytes_to_tmp(file_bytes, name):
    suffix = os.path.splitext(name)[-1].lower() or ".jpg"
    t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    t.write(file_bytes); t.close()
    return t.name

@st.cache_data(show_spinner=False, max_entries=20)
def cached_exif(file_bytes, name):
    tp = _bytes_to_tmp(file_bytes, name)
    try: return extract_exif(tp)
    finally:
        try: os.remove(tp)
        except Exception: pass

@st.cache_data(show_spinner=False, max_entries=20)
def cached_vision(file_bytes, name):
    return analyse_image_from_bytes(file_bytes, name)

@st.cache_data(show_spinner=False, max_entries=20)
def cached_enhanced_ocr(file_bytes, name):
    tp = _bytes_to_tmp(file_bytes, name)
    try: return enhanced_ocr(tp)
    finally:
        try: os.remove(tp)
        except Exception: pass

@st.cache_data(show_spinner=False, max_entries=20)
def cached_identity(file_bytes, name, _vision):
    tp = _bytes_to_tmp(file_bytes, name)
    try: return identify_people_via_vision(tp, vision_context=_vision)
    finally:
        try: os.remove(tp)
        except Exception: pass

@st.cache_data(show_spinner=False, max_entries=50)
def cached_place(lat, lon):
    return place_name(lat, lon)

@st.cache_data(show_spinner=False, max_entries=50)
def cached_corroborate(name, role=""):
    return corroborate_identity(name, role)

@st.cache_data(show_spinner=False, max_entries=20)
def cached_report(file_bytes, name, risk_level, _exif, _vision, _identity_matches):
    return generate_report(exif_data=_exif, vision_data=_vision,
                           identity_matches=_identity_matches, filename=name,
                           risk_level=risk_level)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Image Analysis Demo")
    st.caption("AI-powered media intelligence & privacy awareness")
    st.divider()

    mode = st.radio(
        "Analysis Mode",
        ["Image Analysis", "Video Analysis", "Bulk Image Scan"],
        index=0
    )

    st.divider()
    st.subheader("Service Status")

    # Vision AI
    endpoint = os.getenv("AZURE_VISION_ENDPOINT", "")
    key = os.getenv("AZURE_VISION_KEY", "")
    if endpoint and key:
        st.success("Vision AI: Connected")
    else:
        st.error("Vision AI: Not configured")

    # AI model — Gemini or Groq; asked whether the key really works (it used to say
    # "Connected" for any key filled in, even a dead one).
    @st.cache_data(ttl=300, show_spinner=False)
    def _llm_status():
        return check_llm()
    llm_ok, llm_msg = _llm_status()
    if llm_ok:
        st.success(f"AI ({PROVIDER}): Connected · {llm_msg}")
    elif PROVIDER:
        st.error(f"AI ({PROVIDER}): {llm_msg}")
    else:
        st.warning("AI: Not configured")

    # Serper
    if os.getenv("SERPER_API_KEY"):
        st.success("Identity Search: Connected")
    else:
        st.warning("Identity Search: Not configured")

    st.divider()
    st.caption("Developed for security awareness demonstrations.")


# ── Page heading ─────────────────────────────────────────────────────────────
st.markdown("## Image Analysis Demo")
st.caption("AI-powered media intelligence & privacy awareness tool")
st.warning("Demo only — please use staged photos, not personal ones. Every image you upload is sent to "
           "Azure (India) and Google for analysis: exactly the kind of sharing this session warns about.")

# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "Image Analysis":

    uploaded = st.file_uploader(
        "Drop image (JPG, PNG, HEIC, WEBP, TIFF, BMP)",
        type=["jpg", "jpeg", "png", "webp", "tiff", "bmp", "heic", "heif"]
    )

    if uploaded:
        uploaded.seek(0)
        file_bytes = uploaded.getvalue()
        tmp_path = save_upload(uploaded)

        with st.spinner("Extracting metadata..."):
            exif = cached_exif(file_bytes, uploaded.name)

        import pandas as pd
        gps = exif.get("gps")
        device = exif.get("device", {})
        dt = exif.get("datetime")
        risk = exif.get("privacy_risk", "LOW")
        info = exif.get("image_info", {})

        # ── ROW 1: Image | Metadata | GPS map — all in one row ──────────────
        has_gps = gps and "latitude" in gps
        if has_gps:
            col_img, col_meta, col_map = st.columns([1, 1, 1])
        else:
            col_img, col_meta = st.columns([1, 1.5])

        with col_img:
            st.image(uploaded, use_container_width=True)
            st.caption(f"{uploaded.name} · {round(uploaded.size/1024, 1)} KB")

        with col_meta:
            st.markdown(f"**Metadata risk:** {risk_badge(risk)}", unsafe_allow_html=True)
            if device:
                st.write(f"**Device:** {device.get('make','')} {device.get('model','')}")
                if device.get("software"):
                    sw = device["software"]
                    # iPhones store just the number ("17.0.1"); label it so the point lands.
                    label = f"iOS {sw}" if sw[:1].isdigit() and device.get("make","").lower().startswith("apple") else sw
                    st.write(f"**Software:** {label}")
            if exif.get("datetime_local"):
                st.write(f"**Captured:** {exif['datetime_local']}")
            elif dt:
                st.write(f"**Captured:** {dt}")
            if has_gps:
                st.code(f"{gps['latitude']}, {gps['longitude']}  alt:{gps.get('altitude_m','?')}m", language=None)
                st.markdown(f"[Google Maps]({gps['google_maps_url']}) · [Satellite]({gps['gmaps_satellite']})")
            elif exif.get("has_exif"):
                st.success("EXIF found — no GPS embedded")
            else:
                st.warning("No EXIF — likely stripped by platform")
            st.write(f"{info.get('width','?')}x{info.get('height','?')}px · {info.get('file_size_kb','?')} KB · {info.get('format','?')}")

        if has_gps:
            with col_map:
                st.map(pd.DataFrame({"lat": [gps["latitude"]], "lon": [gps["longitude"]]}), zoom=14, height=220)
            with st.spinner("Resolving the coordinates to a real place..."):
                place = cached_place(gps["latitude"], gps["longitude"])
            if place.get("short"):
                gps["place"] = place.get("display_name") or place["short"]
                st.error(f"📍 **These coordinates resolve to:** {place['short']}  \n"
                         f"<small>{place.get('display_name','')}</small>", icon="📍")
                st.markdown(f"<small>{place.get('display_name','')}</small>", unsafe_allow_html=True)

        # ── Download a CLEANED copy (the fix: strip every hidden tag) ──────────
        try:
            from PIL import Image as _PILImage
            import io as _io
            _clean = _PILImage.open(tmp_path).convert("RGB")
            _buf = _io.BytesIO()
            _clean.save(_buf, format="JPEG", quality=92)  # re-encoding drops EXIF/GPS
            st.download_button(
                "⬇️ Download a clean copy (location & metadata removed)",
                data=_buf.getvalue(),
                file_name=f"CLEANED_{os.path.splitext(uploaded.name)[0]}.jpg",
                mime="image/jpeg",
                help="This is the safe version to share — all GPS and device tags are gone."
            )
        except Exception:
            pass

        # ── Headline: what a stranger learns from THIS one photo ──────────────
        reveals = []
        if has_gps and gps.get("place"):
            reveals.append(f"📍 **Exactly where you were** — {gps['place'].split(',')[0]}" +
                           (f", {gps['place'].split(',')[1].strip()}" if ',' in gps['place'] else ""))
        elif has_gps:
            reveals.append(f"📍 **Exact coordinates** — {gps['latitude']}, {gps['longitude']}")
        if exif.get("datetime_local"):
            reveals.append(f"🕒 **When you were there** — {exif['datetime_local']}")
        if device.get("model"):
            sw = device.get("software", "")
            swlabel = f", iOS {sw}" if sw[:1].isdigit() and device.get('make','').lower().startswith('apple') else (f" ({sw})" if sw else "")
            reveals.append(f"📱 **Your device** — {device.get('make','')} {device.get('model','')}{swlabel}")
        if reveals:
            st.markdown("#### 🔎 What this one photo just told a stranger")
            st.error("  \n".join(reveals))

        # Raw EXIF (collapsed)
        raw = exif.get("raw_tags", {})
        if raw:
            with st.expander("Raw EXIF Tags"):
                for k, v in raw.items():
                    st.text(f"{k}: {v}")

        # ── Vision AI Analysis ────────────────────────────────────────────
        st.divider()
        vision = {}
        if not os.getenv("AZURE_VISION_KEY"):
            st.error("Vision AI not configured in .env")
        else:
            with st.spinner("Running Vision AI..."):
                vision = cached_vision(file_bytes, uploaded.name)

            if vision.get("errors"):
                for err in vision["errors"]:
                    st.error(err)

            # Scene + tags + faces + objects — all in one compact row
            if vision.get("description"):
                st.markdown(f"**Scene:** {vision['description']} *({vision['confidence']}%)*")

            col_v1, col_v2, col_v3 = st.columns(3)

            with col_v1:
                tags = vision.get("tags", [])
                if tags:
                    st.markdown("**Tags:** " + " · ".join([f"`{t['name']}`" for t in tags[:10]]))
                landmarks = vision.get("landmarks", [])
                if landmarks:
                    for lm in landmarks:
                        st.markdown(f"**Landmark:** {lm['name']} {risk_badge('HIGH')}", unsafe_allow_html=True)

            with col_v2:
                faces = vision.get("faces", [])
                if faces:
                    st.markdown(f"**Faces:** {len(faces)} detected — each one reverse-searchable {risk_badge('HIGH')}", unsafe_allow_html=True)
                objects = vision.get("objects", [])
                if objects:
                    from collections import Counter as _Counter
                    _oc = _Counter(o['object'] for o in objects)
                    st.markdown("**Objects:** " + ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in _oc.items()))

            with col_v3:
                brands = vision.get("brands", [])
                if brands:
                    st.markdown("**Brands:** " + ", ".join([b['brand'] for b in brands]))
                cats = vision.get("categories", [])
                if cats:
                    st.markdown("**Categories:** " + ", ".join([c['name'] for c in cats[:5]]))

            # ── OCR — enhanced or basic, collapsed ───────────────────────────
            ocr_lines = vision.get("ocr_text", [])
            enhanced_ocr_result = None

            if vision and should_enhanced_ocr(vision):
                with st.spinner("Enhanced OCR..."):
                    enhanced_ocr_result = cached_enhanced_ocr(file_bytes, uploaded.name)
                if enhanced_ocr_result and not enhanced_ocr_result.get("error"):
                    blocks = enhanced_ocr_result.get("text_blocks", [])
                    full = enhanced_ocr_result.get("full_text", "")
                    words = enhanced_ocr_result.get("words", [])
                    n_words = len(words)
                    with st.expander(f"Extracted Text — {n_words} words read off the image · 🔴 HIGH", expanded=(n_words <= 40)):
                        for block in blocks:
                            st.code(block["text"], language=None)
                        if n_words:
                            avg_conf = sum(w.get("confidence", 0) for w in words) / n_words
                            st.caption(f"{n_words} words · {avg_conf:.0%} avg confidence")
            elif ocr_lines:
                with st.expander(f"Extracted Text — {len(ocr_lines)} lines · {'🔴 HIGH' if len(ocr_lines) > 2 else '🟠 MEDIUM'}", expanded=(len(ocr_lines) <= 8)):
                    for line in ocr_lines:
                        st.code(line, language=None)

        # ── Identity & Context Recognition ──────────────────────────────────
        identity_result = None
        if vision and LLM_KEY:
            st.divider()
            st.subheader("Context & Personnel Analysis")

            with st.spinner("Vision AI scanning for badges, insignia, banners, text..."):
                identity_result = cached_identity(file_bytes, uploaded.name, vision)

            if identity_result.get("error"):
                st.warning(identity_result["error"])
            else:
                # Public figures — grounded in readable text, then searched by NAME on the web.
                figs = identity_result.get("public_figures", [])
                if figs:
                    st.markdown("**🌐 Public figure(s) identified**")
                    for fig in figs:
                        nm = fig.get("name", "")
                        if not nm:
                            continue
                        st.markdown(f'<div class="identity-card">**{nm}** — {fig.get("note","")}  \n'
                                    f'<small>Basis: {fig.get("basis","")}</small></div>', unsafe_allow_html=True)
                        if os.getenv("SERPER_API_KEY"):
                            with st.spinner(f"Searching the web for: {nm}"):
                                corr = cached_corroborate(nm, fig.get("note", ""))
                            news = corr.get("news", [])[:3]
                            if news:
                                for item in news:
                                    st.markdown(f"- [{item['title']}]({item['link']})")
                            ref_imgs = corr.get("images", [])[:3]
                            if ref_imgs:
                                cols = st.columns(len(ref_imgs))
                                for idx, img in enumerate(ref_imgs):
                                    with cols[idx]:
                                        try:
                                            st.image(img["imageUrl"], use_container_width=True)
                                        except Exception:
                                            pass
                    st.caption("Grounded in text/public status — not face-matching of private people.")

                col_id1, col_id2 = st.columns(2)

                with col_id1:
                    # People
                    people = identity_result.get("people_identified", [])
                    if people:
                        st.markdown(f"**Personnel ({len(people)})**")
                        for person in people:
                            badge_text = person.get("name_tag_text")
                            uniform = person.get("uniform_details")
                            desc = person.get("description", "")
                            role = person.get("role_inference", "")

                            st.markdown(f'<div class="identity-card">', unsafe_allow_html=True)
                            if badge_text and badge_text != "null":
                                st.markdown(f"**Badge/Tag:** `{badge_text}`")
                                st.markdown(risk_badge("HIGH"), unsafe_allow_html=True)
                            if uniform and uniform != "null":
                                st.write(f"**Uniform:** {uniform}")
                            if desc:
                                st.write(desc)
                            if role:
                                st.caption(f"Inferred: {role}")
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Corroborate names found on badges
                            if badge_text and badge_text != "null" and os.getenv("SERPER_API_KEY"):
                                with st.spinner(f"Searching: {badge_text}"):
                                    corr = cached_corroborate(badge_text, role)
                                kg = corr.get("knowledge_graph")
                                if kg and kg.get("title"):
                                    st.write(f"**Match:** {kg['title']} — {kg.get('type','')}")
                                    if kg.get("description"):
                                        st.caption(kg["description"][:150])
                                ref_imgs = corr.get("images", [])[:3]
                                if ref_imgs:
                                    cols = st.columns(len(ref_imgs))
                                    for idx, img in enumerate(ref_imgs):
                                        with cols[idx]:
                                            try:
                                                st.image(img["imageUrl"], use_container_width=True)
                                            except Exception:
                                                pass

                with col_id2:
                    # Event
                    event = identity_result.get("event_identified")
                    if event and event.get("type"):
                        st.markdown("**Event**")
                        st.write(f"- {event.get('type', '?')}: {event.get('name', event.get('description', ''))}")
                        if event.get("date"):
                            st.write(f"- Date: {event['date']}")

                    # Venue
                    venue = identity_result.get("venue_identified")
                    if venue and venue.get("type"):
                        st.markdown("**Venue**")
                        st.write(f"- {venue.get('type', '')} {('— ' + venue['name']) if venue.get('name') else ''}")
                        if venue.get("country_indicators"):
                            st.write(f"- Country clues: {venue['country_indicators']}")

                    # Flags & emblems
                    flags = identity_result.get("flags_and_emblems", [])
                    if flags:
                        st.markdown("**Flags / Emblems**")
                        for f in flags:
                            st.write(f"- {f}")

                # Text found by vision LLM
                texts = identity_result.get("text_found", [])
                if texts:
                    with st.expander(f"All Text Read by Vision AI ({len(texts)} fragments)"):
                        for t in texts:
                            st.code(t, language=None)

                # Corroborate event via Serper
                event = identity_result.get("event_identified")
                if event and event.get("name") and os.getenv("SERPER_API_KEY"):
                    with st.spinner(f"Searching: {event['name']}"):
                        corr = cached_corroborate(event["name"])
                    news = corr.get("news", [])
                    if news:
                        st.markdown(f"**Event Coverage ({len(news)} results)**")
                        for item in news[:4]:
                            st.markdown(f"- [{item['title']}]({item['link']})")

                # Key observations
                extra = identity_result.get("additional_observations", "")
                if extra:
                    st.info(extra)

        # ── LLM Intelligence Briefing ────────────────────────────────────────
        if vision and LLM_KEY:
            st.divider()
            st.subheader("Intelligence Briefing")

            identity_matches = []
            if identity_result:
                for p in identity_result.get("people_identified", []):
                    name = p.get("name_tag_text") or p.get("description", "")
                    identity_matches.append({"title": name, "link": "#", "snippet": p.get("role_inference", "")})

            with st.spinner("Generating briefing..."):
                # Overall exposure = worst of metadata risk and content risk.
                _order = {"LOW": 0, "LOW-MEDIUM": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
                overall = risk
                _content_high = bool(vision.get("faces")) or len(vision.get("ocr_text", []) or []) >= 3 \
                    or (identity_result and (identity_result.get("public_figures") or identity_result.get("people_identified")))
                if _content_high and _order.get(overall, 0) < _order["HIGH"]:
                    overall = "HIGH"
                report = cached_report(file_bytes, uploaded.name, overall, exif, vision, identity_matches)
            st.markdown(report)

        # ── Image Chatbot ────────────────────────────────────────────────────
        if LLM_KEY:
            st.divider()
            st.subheader("Ask About This Image")
            st.caption("Chat with AI about the uploaded image — ask about people, objects, text, context, or anything visible")

            # Build analysis context for the chatbot
            analysis_summary_parts = []
            if exif.get("gps") and "latitude" in exif.get("gps", {}):
                g = exif["gps"]
                analysis_summary_parts.append(f"GPS: {g['latitude']}, {g['longitude']}")
            if exif.get("device"):
                d = exif["device"]
                analysis_summary_parts.append(f"Device: {d.get('make','')} {d.get('model','')}")
            if exif.get("datetime"):
                analysis_summary_parts.append(f"Captured: {exif['datetime']}")
            if vision.get("description"):
                analysis_summary_parts.append(f"Scene: {vision['description']}")
            if vision.get("ocr_text"):
                analysis_summary_parts.append(f"OCR text: {'; '.join(vision['ocr_text'][:10])}")
            if vision.get("faces"):
                analysis_summary_parts.append(f"Faces detected: {len(vision['faces'])}")
            if vision.get("landmarks"):
                analysis_summary_parts.append(f"Landmarks: {', '.join(l['name'] for l in vision['landmarks'])}")
            if identity_result and identity_result.get("people_identified"):
                ppl = ", ".join(p.get("name", "?") for p in identity_result["people_identified"])
                analysis_summary_parts.append(f"People identified: {ppl}")

            analysis_ctx = "\n".join(analysis_summary_parts)

            # Session state for chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "chat_image_name" not in st.session_state:
                st.session_state.chat_image_name = None

            # Reset chat if different image uploaded
            if st.session_state.chat_image_name != uploaded.name:
                st.session_state.chat_history = []
                st.session_state.chat_image_name = uploaded.name

            # Display chat history
            for msg in st.session_state.chat_history:
                role = msg["role"]
                with st.chat_message("user" if role == "user" else "assistant"):
                    st.write(msg["content"])

            # Chat input
            user_q = st.chat_input("Ask anything about this image...")
            if user_q:
                # Show user message
                with st.chat_message("user"):
                    st.write(user_q)

                # Build conversation history for multi-turn
                conv_history = []
                for msg in st.session_state.chat_history:
                    conv_history.append({"role": msg["role"], "content": msg["content"]})

                with st.chat_message("assistant"):
                    with st.spinner("Analysing..."):
                        answer = chat_about_image(
                            file_path=tmp_path,
                            user_question=user_q,
                            conversation_history=conv_history,
                            analysis_context=analysis_ctx
                        )
                    st.write(answer)

                # Save to history
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Keep tmp_path alive for chatbot — will be cleaned up on next upload
        if "current_tmp_path" in st.session_state and st.session_state.current_tmp_path != tmp_path:
            try:
                os.remove(st.session_state.current_tmp_path)
            except Exception:
                pass
        st.session_state.current_tmp_path = tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — VIDEO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Video Analysis":
    st.title("Video Analysis")
    st.markdown("Upload a video to extract keyframes, metadata (including embedded GPS), and run scene intelligence on each frame.")

    uploaded = st.file_uploader(
        "Drop video here (MP4, MOV, AVI, MKV, 3GP)",
        type=["mp4", "mov", "avi", "mkv", "3gp", "webm"]
    )

    max_frames = st.slider("Max keyframes to analyse", min_value=4, max_value=20, value=8)

    if uploaded:
        tmp_path = save_upload(uploaded)

        st.subheader("Video Metadata")
        with st.spinner("Extracting video metadata..."):
            vmeta = get_video_metadata(tmp_path)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Duration", f"{vmeta.get('duration_sec', '?')} sec")
        col_m2.metric("Resolution", vmeta.get("resolution", "?"))
        col_m3.metric("FPS", vmeta.get("fps", "?"))
        col_m4.metric("File Size", f"{vmeta.get('file_size_mb', '?')} MB")

        if vmeta.get("codec"):
            st.write(f"**Codec:** {vmeta['codec']}")
        if vmeta.get("creation_time"):
            st.write(f"**Created:** {vmeta['creation_time']}")

        if vmeta.get("location"):
            loc = vmeta["location"]
            st.divider()
            st.subheader("Embedded GPS in Video")
            st.markdown(risk_badge("CRITICAL"), unsafe_allow_html=True)
            if "latitude" in loc:
                c1, c2 = st.columns(2)
                with c1:
                    st.code(f"Latitude:  {loc['latitude']}\nLongitude: {loc['longitude']}\nAltitude:  {loc.get('altitude_m','N/A')} m")
                    st.markdown(f"[Open in Google Maps]({loc['google_maps_url']})")
                with c2:
                    import pandas as pd
                    st.map(pd.DataFrame({"lat": [loc["latitude"]], "lon": [loc["longitude"]]}), zoom=13)
            else:
                st.write(loc.get("raw", ""))

        st.divider()
        st.subheader("Frame-by-Frame Analysis")

        if not os.getenv("AZURE_VISION_KEY"):
            st.error("Vision AI key not configured.")
        else:
            with st.spinner(f"Extracting {max_frames} keyframes and analysing with Vision AI..."):
                video_result = analyse_video(tmp_path, max_frames=max_frames)

            st.success(f"Analysed {video_result['frames_analysed']} keyframes")

            agg = video_result.get("aggregated", {})

            all_tags = agg.get("all_tags", {})
            if all_tags:
                st.markdown("**Top Tags Across All Frames**")
                top_tags = list(all_tags.items())[:20]
                tag_str = " · ".join([f"`{k}` ({v}%)" for k, v in top_tags])
                st.markdown(tag_str)

            landmarks = agg.get("all_landmarks", [])
            if landmarks:
                st.markdown("**Landmarks Identified**")
                st.markdown(risk_badge("HIGH"), unsafe_allow_html=True)
                for lm in landmarks:
                    st.success(f"{lm['name']} — {lm['confidence']}%")

            ocr = agg.get("all_ocr_text", [])
            if ocr:
                st.markdown("**Text Extracted Across Frames**")
                st.markdown(risk_badge("HIGH"), unsafe_allow_html=True)
                for line in ocr:
                    st.code(line)

            faces = agg.get("all_faces", [])
            if faces:
                st.markdown(f"**Total Face Detections: {len(faces)}** across all frames")
                st.markdown(risk_badge("HIGH"), unsafe_allow_html=True)

            obj_counts = agg.get("all_objects", {})
            if obj_counts:
                st.markdown("**Objects Detected (across all frames)**")
                obj_str = ", ".join([f"{k} x{v}" for k, v in sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)])
                st.write(obj_str)

            # LLM report on video
            if LLM_KEY:
                st.divider()
                st.subheader("AI Intelligence Briefing")
                # Build a synthetic vision dict from aggregated
                synth_vision = {
                    "description": f"Video with {video_result['frames_analysed']} frames analysed",
                    "confidence": None,
                    "tags": [{"name": k, "confidence": v} for k, v in list(all_tags.items())[:15]],
                    "objects": [{"object": k, "confidence": 0} for k in obj_counts],
                    "landmarks": landmarks,
                    "faces": faces,
                    "ocr_text": ocr,
                    "brands": agg.get("all_brands", []),
                    "categories": []
                }
                synth_exif = {"has_exif": True, "gps": vmeta.get("location"), "device": {}, "datetime": vmeta.get("creation_time")}
                with st.spinner("Generating video intelligence report..."):
                    report = generate_report(synth_exif, synth_vision, filename=uploaded.name)
                st.markdown(report)

            st.divider()
            st.subheader("Per-Frame Details")
            frames = video_result.get("frame_results", [])

            for i, frame in enumerate(frames):
                with st.expander(f"Frame at {frame['timestamp_sec']}s"):
                    col_fi, col_fd = st.columns([1, 1])
                    with col_fi:
                        if os.path.exists(frame["frame_path"]):
                            st.image(frame["frame_path"], use_container_width=True)
                    with col_fd:
                        if frame.get("description"):
                            st.write(f"**Scene:** {frame['description']} ({frame['confidence']}%)")
                        frame_faces = frame.get("faces", [])
                        if frame_faces:
                            st.write(f"**Faces:** {len(frame_faces)}")
                        frame_lm = frame.get("landmarks", [])
                        if frame_lm:
                            for lm in frame_lm:
                                st.success(f"Landmark: {lm['name']}")
                        frame_ocr = frame.get("ocr_text", [])
                        if frame_ocr:
                            st.write("**Text:**")
                            for line in frame_ocr:
                                st.code(line)

        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — BULK IMAGE SCAN
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Bulk Image Scan":
    st.title("Bulk Image Scan")
    st.markdown("Upload multiple images at once. Get a summary table of findings — GPS, faces, text, landmarks.")

    uploaded_files = st.file_uploader(
        "Drop multiple images (JPG, PNG, HEIC supported)",
        type=["jpg", "jpeg", "png", "webp", "tiff", "heic", "heif"],
        accept_multiple_files=True
    )

    run_vision = st.checkbox("Run Vision AI on each image (slower but more thorough)", value=True)

    if uploaded_files and st.button("Analyse All", type="primary"):
        import pandas as pd

        rows = []
        progress = st.progress(0)
        status_text = st.empty()

        for i, f in enumerate(uploaded_files):
            status_text.text(f"Processing {f.name} ({i+1}/{len(uploaded_files)})...")
            row = {
                "Filename": f.name,
                "Size (KB)": round(f.size / 1024, 1),
                "Has EXIF": False,
                "GPS": "No",
                "Latitude": None,
                "Longitude": None,
                "Device": "",
                "Captured": "",
                "Risk": "LOW",
                "Faces": 0,
                "Landmarks": "",
                "OCR Text": "",
                "Scene": "",
                "Maps Link": "",
                "Error": ""
            }

            tmp = None
            try:
                f.seek(0)
                tmp = save_upload(f)

                exif = extract_exif(tmp)
                row["Has EXIF"] = exif.get("has_exif", False)
                row["Risk"] = exif.get("privacy_risk", "LOW")

                gps = exif.get("gps")
                if gps and "latitude" in gps:
                    row["GPS"] = "YES"
                    row["Latitude"] = gps["latitude"]
                    row["Longitude"] = gps["longitude"]
                    row["Maps Link"] = gps["google_maps_url"]

                device_info = exif.get("device", {})
                row["Device"] = f"{device_info.get('make','')} {device_info.get('model','')}".strip()
                row["Captured"] = exif.get("datetime", "") or ""

                if run_vision and os.getenv("AZURE_VISION_KEY"):
                    try:
                        with open(tmp, "rb") as fb:
                            vis = analyse_image_from_bytes(fb.read(), f.name)
                        row["Faces"] = len(vis.get("faces", []))
                        lm = vis.get("landmarks", [])
                        row["Landmarks"] = ", ".join([l["name"] for l in lm]) if lm else ""
                        ocr = vis.get("ocr_text", [])
                        row["OCR Text"] = " | ".join(ocr[:3]) if ocr else ""
                        row["Scene"] = vis.get("description", "") or ""
                        if row["Risk"] == "LOW" and (row["Faces"] > 0 or row["Landmarks"]):
                            row["Risk"] = "MEDIUM"
                    except Exception as ve:
                        row["Error"] = f"Vision: {str(ve)[:80]}"
            except Exception as e:
                row["Error"] = str(e)[:120]
            finally:
                if tmp:
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

            rows.append(row)
            progress.progress((i + 1) / len(uploaded_files))

        status_text.text("Done.")
        progress.empty()

        df = pd.DataFrame(rows)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Images", len(df))
        col_s2.metric("With GPS", int((df["GPS"] == "YES").sum()))
        col_s3.metric("With EXIF", int(df["Has EXIF"].sum()))
        critical_count = int((df["Risk"] == "CRITICAL").sum())
        col_s4.metric("Critical Risk", critical_count, delta=None)

        st.divider()
        st.subheader("Results Table")

        def colour_risk(val):
            colours = {"CRITICAL": "background-color:#ff4b4b;color:white",
                       "HIGH": "background-color:#ff8c00;color:white",
                       "MEDIUM": "background-color:#ffd700",
                       "LOW": "background-color:#28a745;color:white"}
            return colours.get(val, "")

        styled = df.style.map(colour_risk, subset=["Risk"])
        st.dataframe(styled, use_container_width=True, height=400)

        gps_df = df[df["GPS"] == "YES"].dropna(subset=["Latitude", "Longitude"])
        if not gps_df.empty:
            st.divider()
            st.subheader(f"GPS Locations Found ({len(gps_df)} images)")
            st.markdown(risk_badge("CRITICAL"), unsafe_allow_html=True)
            map_df = gps_df.rename(columns={"Latitude": "lat", "Longitude": "lon"})
            st.map(map_df[["lat", "lon"]], zoom=5)

            for _, r in gps_df.iterrows():
                st.write(f"- **{r['Filename']}** -> [{r['Latitude']}, {r['Longitude']}]({r['Maps Link']})")

        csv = df.to_csv(index=False)
        st.download_button("Download Results CSV", csv, "osint_scan_results.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.72rem; '
    'padding:10px 16px; margin-top:18px; border-top:1px solid #ddd; '
    'background:#fafafa; border-radius:4px; line-height:1.5;">'
    '&copy; 2026 <strong>Yatish Chaturvedi</strong> &nbsp;&middot;&nbsp; '
    'OSINT Image Analyser<br/>'
    '<em>Disclaimer: This application is intended for demonstration purposes only.</em>'
    '</div>',
    unsafe_allow_html=True
)
