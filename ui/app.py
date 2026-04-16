import base64
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"
COMPLETE_API_URL = "http://127.0.0.1:8000/predict_from_features"
IMAGE_API_URL = "http://127.0.0.1:8000/generate-image"

st.set_page_config(
    page_title="Estima · AI Property Valuator",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

FIELD_LABELS = {
    "OverallQual": "Overall Quality",
    "GrLivArea": "Living Area (sq ft)",
    "Neighborhood": "Neighborhood",
    "TotalBsmtSF": "Basement Size (sq ft)",
    "GarageCars": "Garage Capacity (cars)",
    "FullBath": "Full Bathrooms",
    "LotArea": "Lot Size (sq ft)",
    "BedroomAbvGr": "Bedrooms",
    "HouseStyle": "House Style",
    "HouseAge": "House Age (years)",
}

FIELD_HELP = {
    "OverallQual": "Rate the overall house quality from 1 to 10.",
    "GrLivArea": "Above-ground living area in square feet.",
    "Neighborhood": "Neighborhood name, such as NAmes or CollgCr.",
    "TotalBsmtSF": "Basement size in square feet.",
    "GarageCars": "How many cars fit in the garage.",
    "FullBath": "Number of full bathrooms only.",
    "LotArea": "Lot size in square feet.",
    "BedroomAbvGr": "Number of bedrooms above ground.",
    "HouseStyle": "Examples: 1Story, 2Story, SLvl, SFoyer.",
    "HouseAge": "Age of the house in years.",
}

FIELD_ICONS = {
    "OverallQual": "★",
    "GrLivArea": "⊞",
    "Neighborhood": "◎",
    "TotalBsmtSF": "⬓",
    "GarageCars": "⬡",
    "FullBath": "◈",
    "LotArea": "⬜",
    "BedroomAbvGr": "⌂",
    "HouseStyle": "⬢",
    "HouseAge": "◷",
}

NUMERIC_FIELDS = {
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars",
    "FullBath", "LotArea", "BedroomAbvGr", "HouseAge",
}


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #0e0e0f;
    color: #f0ede6;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3rem 4rem 3rem;
    max-width: 1200px;
}

.wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #c9a96e;
}

.hero-headline {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.1;
    color: #f0ede6;
    margin: 0.3rem 0 0.6rem 0;
}

.hero-sub {
    font-size: 0.95rem;
    font-weight: 300;
    color: #8a867c;
    letter-spacing: 0.01em;
    margin-bottom: 2.5rem;
}

.rule {
    height: 1px;
    background: linear-gradient(90deg, #c9a96e33, #c9a96e66, #c9a96e33);
    margin: 0 0 2rem 0;
    border: none;
}

.panel {
    background: #161618;
    border: 1px solid #2a2a2d;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}

.panel-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c9a96e;
    margin-bottom: 1rem;
}

/* ── Prompt guide examples ── */
.prompt-example {
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.55rem;
    font-size: 0.84rem;
    line-height: 1.6;
}
.prompt-good {
    background: #0d1f12;
    border: 1px solid #1e4a28;
    color: #a8d8b0;
}
.prompt-bad {
    background: #1f0e0e;
    border: 1px solid #4a1e1e;
    color: #d8a8a8;
}
.prompt-tag {
    display: inline-block;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 4px;
    padding: 0.12rem 0.45rem;
    margin-bottom: 0.35rem;
}
.tag-good { background: #1e4a28; color: #6fcf97; }
.tag-bad  { background: #4a1e1e; color: #eb5757; }

/* ── Widgets ── */
.stTextArea textarea {
    background: #0e0e0f !important;
    border: 1px solid #2e2e32 !important;
    border-radius: 10px !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.7 !important;
    padding: 0.9rem 1rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #c9a96e !important;
    box-shadow: 0 0 0 3px #c9a96e18 !important;
}

.stTextInput input {
    background: #0e0e0f !important;
    border: 1px solid #2e2e32 !important;
    border-radius: 8px !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s;
}
.stTextInput input:focus {
    border-color: #c9a96e !important;
    box-shadow: 0 0 0 3px #c9a96e18 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #c9a96e, #a07840) !important;
    color: #0e0e0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

.stTextArea label, .stTextInput label {
    color: #8a867c !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}

/* ── Radio pill selector ── */
.stRadio > div {
    gap: 0.5rem !important;
    flex-wrap: wrap !important;
}
.stRadio > div > label {
    background: #1a1a1d !important;
    border: 1px solid #2e2e32 !important;
    border-radius: 50px !important;
    padding: 0.4rem 0.95rem !important;
    font-size: 0.84rem !important;
    color: #8a867c !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
}
.stRadio > div > label:hover {
    border-color: #c9a96e88 !important;
    color: #c9a96e !important;
}
.stRadio > div > label[data-baseweb="radio"] input:checked + div,
div[data-testid="stRadio"] label[aria-checked="true"] {
    border-color: #c9a96e !important;
    background: #1e1810 !important;
    color: #c9a96e !important;
}
div[data-testid="stRadio"] { margin-bottom: 0 !important; }
div[data-testid="stRadio"] label { margin-bottom: 0 !important; }

/* ── Feature rows ── */
.feat-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid #1e1e21;
}
.feat-row:last-child { border-bottom: none; }
.feat-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #1e1e21;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #c9a96e;
    font-size: 0.95rem;
    flex-shrink: 0;
}
.feat-label {
    font-size: 0.78rem;
    color: #5e5a54;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.feat-value {
    font-size: 0.95rem;
    color: #f0ede6;
    font-weight: 400;
}

/* ── Stat pills ── */
.stat-pill {
    display: flex;
    flex-direction: column;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    background: #161618;
    border: 1px solid #2a2a2d;
    gap: 0.2rem;
}
.stat-pill-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5e5a54;
    font-weight: 500;
}
.stat-pill-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    color: #c9a96e;
    font-weight: 700;
}

/* ── Interpretation ── */
.interp-block {
    font-size: 0.93rem;
    line-height: 1.85;
    color: #b0ac9f;
    font-weight: 300;
    border-left: 2px solid #c9a96e44;
    padding-left: 1.2rem;
    margin-top: 0.8rem;
}

/* ── Badges ── */
.badge-complete {
    display: inline-block;
    background: #1a2a1a;
    color: #6fcf97;
    border: 1px solid #2d5a2d;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
}
.badge-incomplete {
    display: inline-block;
    background: #2a1f0e;
    color: #f2994a;
    border: 1px solid #5a3a0e;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
}

.missing-header {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f2994a;
    margin-bottom: 1.2rem;
}

/* ── Image preview ── */
.img-question {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #f0ede6;
    margin-bottom: 0.3rem;
}
.img-sub {
    font-size: 0.84rem;
    color: #5e5a54;
    margin-bottom: 1.1rem;
}
.img-rendered {
    border-radius: 14px;
    overflow: hidden;
    margin-top: 1.2rem;
    border: 1px solid #2a2a2d;
}

.stSpinner > div { border-top-color: #c9a96e !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="wordmark">⬡ &nbsp; Estima</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-headline">Property Valuation,<br>Reimagined.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Describe any property in plain language — our AI extracts the details and delivers a market-calibrated estimate.</div>', unsafe_allow_html=True)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

if "query" not in st.session_state:
    st.session_state.query = ""

# ─── Input + Prompt Guide (equal vertical alignment, no gap) ──────────────────
col_left, col_right = st.columns([3, 1.6], gap="large")

with col_left:
    query = st.text_area(
        "Property description",
        value=st.session_state.query,
        height=300,
        placeholder="e.g. A 3 bedroom 2-story home, 2 bathrooms, 2-car garage, 2000 sqft living area, 900 sqft basement, built in 2010, NAmes neighborhood, lot size 8000 sqft…",
        label_visibility="collapsed"
    )
    st.markdown("""
<div style="
    margin-top: 0.6rem;
    padding: 0.7rem 0.9rem;
    border-radius: 10px;
    background: #141416;
    border: 1px solid #2a2a2d;
    font-size: 0.84rem;
    color: #8a867c;
    line-height: 1.6;
    margin-bottom: 1.2rem;
">
💡 Include bedrooms, bathrooms, square footage, garage, basement, year built, neighborhood, and lot size for best results.
</div>
""", unsafe_allow_html=True)
    st.session_state.query = query
    analyze = st.button("→  Analyze & Extract", use_container_width=True)

with col_right:
    # Compact prompt guide — no bottom padding panel, all inline
    st.markdown("""
<div style="background:#161618;border:1px solid #2a2a2d;border-radius:16px;padding:1.2rem 1.5rem;">
<div class="panel-label">Prompt Guide</div>
<div class="prompt-example prompt-good">
<span class="prompt-tag tag-good">✓ Good</span><br>
"3 bed 2-story, 2 bath, 2-car garage, 2000 sqft, 900 sqft basement, built 2010, NAmes, lot 8000 sqft."
</div>
<div class="prompt-example prompt-good">
<span class="prompt-tag tag-good">✓ Good</span><br>
"4 bed, 3 bath, 3-car garage, 2400 sqft, CollgCr, 2015, lot 10000 sqft, quality 8/10."
</div>
<div class="prompt-example prompt-bad">
<span class="prompt-tag tag-bad">✗ Vague</span><br>
"A nice house with a garage." — no numbers.
</div>
<div class="prompt-example prompt-bad" style="margin-bottom:0;">
<span class="prompt-tag tag-bad">✗ Incomplete</span><br>
"3 bedrooms." — missing size, age, style.
</div>
</div>
""", unsafe_allow_html=True)

# ─── API call ─────────────────────────────────────────────────────────────────
if analyze:
    if not query.strip():
        st.error("Please enter a property description.")
        st.stop()
    with st.spinner("Extracting property details…"):
        response = requests.post(API_URL, json={"query": query})
    if response.status_code != 200:
        st.error("The API returned an error. Please try again.")
        st.stop()
    st.session_state["api_response"] = response.json()
    for key in ("final_result", "generated_image"):
        st.session_state.pop(key, None)

# ─── Results ──────────────────────────────────────────────────────────────────
if "api_response" in st.session_state:
    data = st.session_state["api_response"]
    status = data["status"]
    features = data["extracted_features"]

    extracted_fields = [(k, FIELD_LABELS.get(k, k), v) for k, v in features.items() if v is not None]
    missing_fields   = [(k, FIELD_LABELS.get(k, k)) for k, v in features.items() if v is None]

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    col_ex, col_sum = st.columns([1.6, 1], gap="large")

    with col_ex:
        rows_html = ""
        if extracted_fields:
            for field_key, label, value in extracted_fields:
                icon = FIELD_ICONS.get(field_key, "·")
                rows_html += f"""
<div class="feat-row">
    <div class="feat-icon">{icon}</div>
    <div>
        <div class="feat-label">{label}</div>
        <div class="feat-value">{value}</div>
    </div>
</div>"""
        else:
            rows_html = '<p style="color:#5e5a54;font-size:0.88rem;margin:0;">No fields were confidently extracted.</p>'

        st.markdown(f"""
<div class="panel">
<div class="panel-label">Extracted Details</div>
{rows_html}
</div>""", unsafe_allow_html=True)

    with col_sum:
        badge = '<span class="badge-complete">Complete</span>' if status == "complete" else '<span class="badge-incomplete">Incomplete</span>'
        st.markdown(f"""
<div class="panel">
<div class="panel-label">Extraction Summary</div>
<div style="margin-bottom:1rem;">{badge}</div>
<div class="feat-row">
    <div class="feat-icon">✓</div>
    <div>
        <div class="feat-label">Extracted</div>
        <div class="feat-value">{len(extracted_fields)} fields</div>
    </div>
</div>
<div class="feat-row">
    <div class="feat-icon" style="color:#f2994a;">?</div>
    <div>
        <div class="feat-label">Missing</div>
        <div class="feat-value">{len(missing_fields)} fields</div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

    # ── Missing fields form ────────────────────────────────────────────────────
    if status == "incomplete":
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown("""
<div class="panel">
<div class="panel-label">Complete Missing Details</div>
<div class="missing-header">Fill in the fields below to generate your estimate.</div>
</div>""", unsafe_allow_html=True)

        user_inputs = {}
        cols = st.columns(2, gap="medium")
        for idx, (field_key, label) in enumerate(missing_fields):
            with cols[idx % 2]:
                user_inputs[field_key] = st.text_input(
                    label,
                    help=FIELD_HELP.get(field_key, ""),
                    key=f"missing_{field_key}"
                )

        submit_completed = st.button("→  Generate Price Estimate", use_container_width=True)

        if submit_completed:
            completed_features = features.copy()
            for field_key, raw_value in user_inputs.items():
                if str(raw_value).strip() == "":
                    st.error(f"Please fill '{FIELD_LABELS[field_key]}'.")
                    st.stop()
                if field_key in NUMERIC_FIELDS:
                    try:
                        completed_features[field_key] = (
                            int(float(raw_value))
                            if field_key in {"OverallQual", "BedroomAbvGr", "HouseAge"}
                            else float(raw_value)
                        )
                    except ValueError:
                        st.error(f"'{FIELD_LABELS[field_key]}' must be a number.")
                        st.stop()
                else:
                    completed_features[field_key] = raw_value

            with st.spinner("Calculating valuation…"):
                response = requests.post(
                    COMPLETE_API_URL,
                    json={"features": completed_features, "query": query}
                )
            if response.status_code != 200:
                st.error("Prediction failed. Please try again.")
                st.stop()
            st.session_state["final_result"] = response.json()
            st.session_state.pop("generated_image", None)

    elif status == "complete":
        st.session_state["final_result"] = data

# ─── Price result ──────────────────────────────────────────────────────────────
if "final_result" in st.session_state:
    result = st.session_state["final_result"]
    prediction = result["prediction"]
    position = prediction["relative_position"].replace("_", " ").title()

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Valuation Result</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
<div class="stat-pill">
<span class="stat-pill-label">Predicted Price</span>
<span class="stat-pill-value">{prediction["formatted_price"]}</span>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="stat-pill">
<span class="stat-pill-label">Typical Range</span>
<span class="stat-pill-value" style="font-size:1.25rem;">${prediction["q1_price"]:,.0f} – ${prediction["q3_price"]:,.0f}</span>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
<div class="stat-pill">
<span class="stat-pill-label">Market Position</span>
<span class="stat-pill-value" style="font-size:1.25rem;">{position}</span>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="panel" style="margin-top:1rem;">
<div class="panel-label">Interpretation</div>
<div class="interp-block">{result["interpretation"]}</div>
</div>""", unsafe_allow_html=True)

    # ─── Image Section ─────────────────────────────────────────────────────────
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    st.markdown("""
<div class="img-question">Want to see what this home looks like?</div>
""", unsafe_allow_html=True)

    generate_clicked = st.button("→  Generate House Preview", use_container_width=True)

    if generate_clicked:
        with st.spinner("Generating house preview…"):
            response = requests.post(
                IMAGE_API_URL,
                json={
                    "query": st.session_state.query,
                    "features": result["extracted_features"],
                }
            )
        if response.status_code != 200:
            st.error("Image generation failed. Please try again.")
            st.stop()
        st.session_state["generated_image"] = response.json()

    if "generated_image" in st.session_state:
        image_data = st.session_state["generated_image"]

        if image_data["status"] == "success" and image_data["image_base64"]:
            image_bytes = base64.b64decode(image_data["image_base64"])
            st.markdown('<div class="img-rendered">', unsafe_allow_html=True)
            st.image(image_bytes, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(image_data["message"])