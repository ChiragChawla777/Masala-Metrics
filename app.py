import os
import json
import html
import random

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# MASALA METRICS | Indian food recognition + nutrition explorer
# ============================================================
st.set_page_config(
    page_title="Masala Metrics — Eat desi. Know better.",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "indian_food_effnetb0.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "food_class_names.json")
NUTRITION_PATH = os.path.join(BASE_DIR, "nutrition.csv")
IMAGE_SIZE = (224, 224)

# A lively Indian editorial palette: indigo ink, haldi, chilli, leaf and paper.
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Serif+Display:ital@0;1&family=Yatra+One&display=swap');
:root{--ink:#201b32;--muted:#776f68;--paper:#fff9ed;--card:#fffdf7;--line:#eadcc5;--indigo:#282343;--indigo2:#3b315e;--haldi:#f4b942;--chilli:#c94e32;--leaf:#456b4b;--pink:#e9a18c;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:#fff9ed;background-image:radial-gradient(rgba(201,78,50,.10) .75px,transparent .75px),linear-gradient(135deg,rgba(244,185,66,.08),transparent 38%);background-size:17px 17px,100% 100%}
.main .block-container{max-width:1320px;padding:1.1rem 2.3rem 3.2rem}
[data-testid="stHeader"]{background:rgba(255,249,237,.92)}
#MainMenu,footer{visibility:hidden}
/* Brand strip */
.brandbar{display:flex;align-items:center;justify-content:space-between;padding:5px 0 17px;border-bottom:1px solid #eadcc5;margin-bottom:23px}
.brandleft{display:flex;align-items:center;gap:12px}.brandseal{height:48px;width:48px;border-radius:16px;background:var(--chilli);color:#fff7e8;display:grid;place-items:center;font-size:26px;box-shadow:4px 4px 0 var(--haldi);transform:rotate(-3deg)}
.brandtitle{font-weight:900;font-size:20px;letter-spacing:-.8px;color:var(--indigo)}.brandtag{font-size:9px;letter-spacing:2px;font-weight:800;color:#a65b3f;margin-top:1px}.brandright{font-size:11px;color:#766d63;font-weight:700;letter-spacing:.7px}
/* Hero */
.hero{position:relative;overflow:hidden;border-radius:28px;background:linear-gradient(118deg,#282343 0%,#3d315d 58%,#8e493c 125%);padding:39px 44px 36px;color:#fff8eb;min-height:280px;box-shadow:0 18px 42px rgba(40,35,67,.17);border:1px solid rgba(255,255,255,.35)}
.hero:before{content:'';position:absolute;inset:0;opacity:.20;background:repeating-linear-gradient(45deg,transparent 0 18px,rgba(255,222,155,.38) 19px,transparent 20px 38px),repeating-linear-gradient(-45deg,transparent 0 18px,rgba(255,222,155,.2) 19px,transparent 20px 38px);mask-image:linear-gradient(90deg,transparent 20%,#000 100%)}
.hero:after{content:'✺';position:absolute;right:5%;top:-70px;font-size:300px;color:#f4b942;opacity:.20;font-family:serif;line-height:1}
.hero-inner{position:relative;z-index:1;max-width:790px}.hero-kicker{display:inline-flex;border:1px solid rgba(255,230,183,.45);background:rgba(255,255,255,.09);padding:7px 13px;border-radius:99px;color:#ffdc91;font-size:10px;font-weight:800;letter-spacing:1.6px;text-transform:uppercase}
.hero h1{font-family:'DM Serif Display',serif!important;font-weight:400!important;color:#fff8eb!important;font-size:clamp(40px,5vw,64px)!important;line-height:1.02!important;letter-spacing:-1.2px!important;margin:17px 0 12px!important}.hero h1 em{color:#f5c45e}.hero p{color:#eee5dc;font-size:15px;line-height:1.75;max-width:650px;margin:0}.hero-foot{margin-top:22px;color:#f6ce7c;font-size:11px;font-weight:800;letter-spacing:1px}
/* Section headings and surfaces */
.eyebrow{color:#b44f36;font-size:10px;font-weight:900;letter-spacing:1.8px;text-transform:uppercase;margin:5px 0 5px}.section-title{font-family:'DM Serif Display',serif;color:var(--indigo);font-size:32px;line-height:1.1;margin:0 0 6px}.section-sub{font-size:13px;color:var(--muted);margin:0 0 17px;line-height:1.65}.panel{background:#fffdf7;border:1px solid var(--line);border-radius:22px;padding:21px 23px;box-shadow:0 8px 22px rgba(69,48,28,.045)}
/* Streamlit controls */
.stButton>button{background:linear-gradient(110deg,#c94e32,#df783b)!important;color:#fffaf0!important;border:0!important;border-radius:13px!important;padding:.78rem 1.1rem!important;font-weight:800!important;box-shadow:0 7px 16px rgba(201,78,50,.20);transition:all .18s ease}.stButton>button:hover{transform:translateY(-2px);filter:brightness(1.06);box-shadow:0 10px 22px rgba(201,78,50,.25)}
.stButton>button:focus{box-shadow:0 0 0 3px rgba(244,185,66,.35)!important}
[data-testid="stFileUploader"]{background:#fffdf7;border:1.5px dashed #d8a65b;border-radius:18px;padding:13px;transition:.2s}[data-testid="stFileUploader"]:hover{background:#fff8e8;border-color:#c94e32}
[data-testid="stSelectbox"],[data-testid="stTextInput"],[data-testid="stSlider"]{background:transparent}
[data-testid="stMetric"]{background:#fffdf7;border:1px solid var(--line);border-radius:16px;padding:14px 16px;box-shadow:0 5px 14px rgba(50,35,20,.04)}[data-testid="stMetricLabel"]{color:#766f67}[data-testid="stMetricValue"]{color:var(--indigo);font-weight:900}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#f4b942,#c94e32)}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:#f5ead7;padding:7px;border-radius:15px}.stTabs [data-baseweb="tab"]{height:43px;border-radius:11px;color:#655b53;font-weight:800;padding:0 18px}.stTabs [aria-selected="true"]{background:#fffdf7!important;color:#a9432d!important;box-shadow:0 3px 10px rgba(45,30,15,.08)}
.stExpander{border:1px solid var(--line)!important;border-radius:15px!important;background:#fffdf7}
[data-testid="stAlert"]{border-radius:14px}
/* Food result */
.result{position:relative;overflow:hidden;background:linear-gradient(110deg,#fff4d9,#fffdf7);border:1px solid #ead4a6;border-left:6px solid var(--haldi);border-radius:21px;padding:22px 25px;margin:8px 0 17px}.result:after{content:'✿';position:absolute;right:20px;top:-23px;font-size:105px;color:#c94e32;opacity:.10}.result-kicker{font-size:10px;letter-spacing:1.6px;text-transform:uppercase;color:#a84b32;font-weight:900}.result-name{font-family:'DM Serif Display',serif;font-size:35px;color:var(--indigo);margin:5px 0}.result-meta{font-size:12px;color:#74685d}
.nutrient{height:100%;background:#fffdf7;border:1px solid var(--line);border-radius:17px;padding:16px 15px;min-height:123px}.nutrient.hot{background:linear-gradient(145deg,#bd4d32,#dd783c);border-color:#bd4d32}.n-icon{font-size:20px}.n-label{font-size:11px;color:#786d62;font-weight:800;margin-top:8px}.n-value{font-size:27px;color:var(--indigo);font-weight:900;letter-spacing:-1px;margin-top:2px}.n-unit{font-size:10px;color:#8b8076}.hot .n-label,.hot .n-unit{color:#ffe3d4}.hot .n-value{color:#fff9ed}
.foodrow{display:flex;justify-content:space-between;gap:12px;font-size:13px;font-weight:800;margin:10px 0 4px}.foodrow span:last-child{color:#b54d32;font-variant-numeric:tabular-nums}
.note{padding:13px 16px;background:#f2ead8;border:1px solid #e6d5b6;border-radius:14px;color:#6e5a40;font-size:12px;line-height:1.7;margin:15px 0}.mini-stat{background:#f7eddb;border:1px solid #ead8b9;border-radius:15px;padding:13px 15px}.mini-stat strong{display:block;color:#302747;font-size:22px}.mini-stat span{color:#786d62;font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase}
.footer{border-top:1px solid #e7d8c2;margin-top:35px;padding:20px 5px 0;text-align:center;color:#8c7d6d;font-size:11px;line-height:1.8}.footer b{color:#44375d}
@media(max-width:750px){.main .block-container{padding-left:1rem;padding-right:1rem}.hero{padding:29px 24px;min-height:260px}.hero:after{right:-16%;top:30%;font-size:170px}.hero p{font-size:13px}.section-title{font-size:27px}.result-name{font-size:28px}.brandright{display:none}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Warming up the kitchen… loading the food model")
def load_food_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data

def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    if isinstance(names, dict):
        names = names.get("class_names", names.get("classes", names))
        if isinstance(names, dict):
            try:
                names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            except (ValueError, TypeError):
                names = list(names.values())
    if not isinstance(names, list):
        raise ValueError("food_class_names.json must contain a list of dish names.")
    return names


@st.cache_data

def load_nutrition_data():
    if not os.path.exists(NUTRITION_PATH):
        raise FileNotFoundError(f"Nutrition file not found: {NUTRITION_PATH}")
    df = pd.read_csv(NUTRITION_PATH)
    required = ["dish_name", "serving_g", "calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"nutrition.csv is missing columns: {missing}")
    df["dish_name"] = df["dish_name"].astype(str).str.strip()
    return df


def display_name(name):
    return str(name).replace("_", " ").title()


def preprocess_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    # EfficientNetB0's embedded preprocessing is preserved from the original app.
    return np.expand_dims(arr, axis=0)


def predict_food(image, model, class_names):
    scores = model.predict(preprocess_image(image), verbose=0)[0]
    if len(scores) != len(class_names):
        raise ValueError(f"Model returns {len(scores)} outputs, but there are {len(class_names)} class names.")
    top = np.argsort(scores)[::-1][:5]
    best = int(top[0])
    return class_names[best], float(scores[best]) * 100, scores, top


def nutrient_card(icon, label, value, unit, hot=False):
    cls = "nutrient hot" if hot else "nutrient"
    st.markdown(
        f'<div class="{cls}"><div class="n-icon">{icon}</div><div class="n-label">{label}</div>'
        f'<div class="n-value">{value}</div><div class="n-unit">{unit}</div></div>',
        unsafe_allow_html=True,
    )


def render_nutrition(row, multiplier=1.0):
    vals = [
        ("🔥", "Calories", row["calories_kcal"], "kcal", True),
        ("💪", "Protein", row["protein_g"], "g protein", False),
        ("🌾", "Carbs", row["carbs_g"], "g carbs", False),
        ("🥑", "Fat", row["fat_g"], "g fat", False),
        ("🥬", "Fibre", row["fiber_g"], "g fibre", False),
    ]
    cols = st.columns(5, gap="small")
    for col, (icon, label, value, unit, hot) in zip(cols, vals):
        with col:
            value_text = "—" if pd.isna(value) else (f"{value * multiplier:.0f}" if label == "Calories" else f"{value * multiplier:.1f}")
            nutrient_card(icon, label, value_text, unit, hot)


def nutrition_for(dish, nutrition):
    found = nutrition[nutrition["dish_name"].str.casefold() == str(dish).strip().casefold()]
    return None if found.empty else found.iloc[0]


def show_scaled_profile(row, key_prefix):
    base = float(row["serving_g"])
    low = max(25, int(round(base * 0.5 / 5) * 5))
    high = max(low + 5, int(round(base * 2.5 / 5) * 5))
    grams = st.slider("Adjust your portion", min_value=low, max_value=high, value=int(base), step=5, key=f"{key_prefix}_portion")
    multiplier = grams / base if base else 1.0
    st.caption(f"Estimated values for **{grams} g** · reference serving: {base:g} g")
    render_nutrition(row, multiplier)
    return grams


# ----------------------------
# Header and intro
# ----------------------------
st.markdown("""
<div class="brandbar">
  <div class="brandleft"><div class="brandseal">🍛</div><div><div class="brandtitle">Masala Metrics</div><div class="brandtag">EAT DESI · KNOW BETTER</div></div></div>
  <div class="brandright">A LITTLE SCIENCE. A LOT OF SWAAD. ✦</div>
</div>
<div class="hero"><div class="hero-inner">
  <div class="hero-kicker">✦ NAMASTE, FOOD EXPLORER</div>
  <h1>Your plate has a story.<br><em>Let’s decode the swaad.</em></h1>
  <p>From ghar ka khaana to your favourite street-food fix — identify an Indian dish, discover its nutrition, and explore 96 familiar flavours with a little help from AI.</p>
  <div class="hero-foot">✺ 96 FOOD CATEGORIES &nbsp; · &nbsp; EFFICIENTNETB0 &nbsp; · &nbsp; NUTRITION EXPLORER</div>
</div></div>
""", unsafe_allow_html=True)

try:
    class_names = load_class_names()
    nutrition_data = load_nutrition_data()
except Exception as exc:
    st.error(f"Couldn't load the food labels or nutrition database: {exc}")
    st.stop()

# Small real project stats, based on files available in this repository.
stat1, stat2, stat3 = st.columns(3)
for col, value, label in [
    (stat1, len(class_names), "FOOD CATEGORIES"),
    (stat2, len(nutrition_data), "NUTRITION ENTRIES"),
    (stat3, "AI", "IMAGE RECOGNITION"),
]:
    with col:
        st.markdown(f'<div class="mini-stat"><strong>{value}</strong><span>{label}</span></div>', unsafe_allow_html=True)

st.write("")
identify_tab, explore_tab, about_tab = st.tabs(["📸  Identify your food", "🥘  Explore the menu", "🪷  How it works"])

# ============================================================
# TAB 1 — IMAGE IDENTIFICATION
# ============================================================
with identify_tab:
    st.markdown('<div class="eyebrow">01 / THE FOOD SCANNER</div><div class="section-title">Aaj kya kha rahe hain?</div><div class="section-sub">Upload one clear photo. We’ll find the closest match from the model’s 96 food categories.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        uploaded = st.file_uploader("Drop your dish photo here", type=["jpg", "jpeg", "png", "webp"], key="food_upload", help="Use a clear JPG, PNG or WEBP image.")
        st.markdown('<div class="note"><b>📷 Pro tips from the rasoi</b><br>• Keep the dish in focus and well lit.<br>• Try to show one main dish at a time.<br>• A top or 45° angle usually works nicely.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="eyebrow">BEFORE YOU SCAN</div><div class="section-title" style="font-size:26px">From photo to food gyaan.</div><p style="color:#776f68;font-size:13px;line-height:1.8">Your image is resized for the trained EfficientNetB0 model. The app then matches the predicted dish with its reference nutrition entry.</p><div style="font-size:12px;color:#a64d34;font-weight:800">✦ No portion weight is detected from the photo.</div></div>', unsafe_allow_html=True)

    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            prev, action = st.columns([1, 1], gap="large")
            with prev:
                st.image(image, caption=f"{uploaded.name} · {image.width} × {image.height} px", use_container_width=True)
            with action:
                st.markdown('<div class="eyebrow">READY WHEN YOU ARE</div><div class="section-title" style="font-size:27px">Shall we take a look?</div>', unsafe_allow_html=True)
                st.write("Hit the button to identify the dish and open its nutrition card.")
                clicked = st.button("✦  Identify this dish", type="primary", use_container_width=True)
                if clicked:
                    try:
                        model = load_food_model()
                        with st.spinner("Chef AI is checking the flavours…"):
                            dish, confidence, scores, top = predict_food(image, model, class_names)
                        st.session_state["food_result"] = {
                            "upload_key": f"{uploaded.name}-{uploaded.size}",
                            "dish": dish,
                            "confidence": confidence,
                            "scores": scores.tolist(),
                            "top": top.tolist(),
                        }
                    except Exception as exc:
                        st.error(f"The scan couldn't be completed: {exc}")

            upload_key = f"{uploaded.name}-{uploaded.size}"
            result = st.session_state.get("food_result")
            if result and result.get("upload_key") == upload_key:
                st.markdown('<div class="eyebrow">02 / THE VERDICT IS IN</div>', unsafe_allow_html=True)
                safe_name = html.escape(display_name(result["dish"]))
                st.markdown(f'<div class="result"><div class="result-kicker">✦ TOP MODEL MATCH</div><div class="result-name">{safe_name}</div><div class="result-meta">Model score · {result["confidence"]:.1f}% &nbsp; | &nbsp; Not a guarantee of correctness</div></div>', unsafe_allow_html=True)
                st.progress(max(0.0, min(1.0, result["confidence"] / 100)))
                row = nutrition_for(result["dish"], nutrition_data)
                if row is None:
                    st.warning("Dish identified, but its nutrition entry is not available yet.")
                else:
                    st.markdown('<div class="section-title" style="font-size:27px">Your plate ka nutrition</div><div class="section-sub">Slide to match the portion you actually ate. Values scale from the reference serving.</div>', unsafe_allow_html=True)
                    show_scaled_profile(row, "scan")
                    st.markdown('<div class="note">ⓘ <b>Nutrition is an estimate.</b> These are reference values scaled by the portion size you select. Actual nutrition changes with recipe, ingredients, oil and preparation. The photo itself does not reveal food weight.</div>', unsafe_allow_html=True)
                with st.expander("👀 Peek at the other top matches"):
                    st.caption("These are the model's five highest output scores, not calibrated probabilities.")
                    scores = np.asarray(result["scores"])
                    for idx in result["top"]:
                        idx = int(idx)
                        score = float(scores[idx]) * 100
                        st.markdown(f'<div class="foodrow"><span>{html.escape(display_name(class_names[idx]))}</span><span>{score:.1f}%</span></div>', unsafe_allow_html=True)
                        st.progress(max(0.0, min(1.0, score / 100)))
        except Exception as exc:
            st.error(f"We couldn't open this image: {exc}")
    else:
        st.markdown('<div class="panel" style="text-align:center;padding:32px 18px"><div style="font-size:38px">🥭 🌶️ 🥘</div><div class="section-title" style="font-size:25px;margin-top:9px">Your next bite starts here.</div><div class="section-sub" style="margin-bottom:0">Upload a food photo above to get your dish identification and nutrition snapshot.</div></div>', unsafe_allow_html=True)

# ============================================================
# TAB 2 — SEARCHABLE NUTRITION EXPLORER
# ============================================================
with explore_tab:
    st.markdown('<div class="eyebrow">02 / THE DESI FOOD DIRECTORY</div><div class="section-title">A menu full of familiar favourites.</div><div class="section-sub">Search the nutrition reference database, pick a dish, and adjust the serving size to see how the estimates change.</div>', unsafe_allow_html=True)
    search_col, sort_col = st.columns([1.5, 1])
    with search_col:
        query = st.text_input("Find a dish", placeholder="Try biryani, dosa, samosa…", key="dish_search")
    with sort_col:
        sort_mode = st.selectbox("Sort menu by", ["A to Z", "Calories: low to high", "Calories: high to low", "Protein: high to low"], key="dish_sort")
    menu = nutrition_data.copy()
    menu["display"] = menu["dish_name"].map(display_name)
    if query.strip():
        mask = menu["display"].str.contains(query.strip(), case=False, na=False) | menu["dish_name"].str.contains(query.strip(), case=False, na=False)
        menu = menu[mask]
    if sort_mode == "Calories: low to high":
        menu = menu.sort_values("calories_kcal", ascending=True)
    elif sort_mode == "Calories: high to low":
        menu = menu.sort_values("calories_kcal", ascending=False)
    elif sort_mode == "Protein: high to low":
        menu = menu.sort_values("protein_g", ascending=False)
    else:
        menu = menu.sort_values("display")
    st.caption(f"Showing **{len(menu)}** of {len(nutrition_data)} nutrition entries.")
    if menu.empty:
        st.info("No dish found. Try a shorter search, like 'dal' or 'rice'.")
    else:
        options = menu["dish_name"].tolist()
        chosen = st.selectbox("Choose a dish to explore", options, format_func=display_name, key="explorer_dish")
        row = nutrition_for(chosen, nutrition_data)
        if row is not None:
            st.markdown(f'<div class="result"><div class="result-kicker">✦ FROM THE MASALA METRICS MENU</div><div class="result-name">{html.escape(display_name(chosen))}</div><div class="result-meta">Reference serving · {float(row["serving_g"]):g} g</div></div>', unsafe_allow_html=True)
            st.markdown("**Build your portion**")
            show_scaled_profile(row, "explorer")
            st.markdown('<div class="note">Nutrition entries are approximate reference values, not laboratory measurements. Recipes and portions vary from kitchen to kitchen.</div>', unsafe_allow_html=True)
        with st.expander("Browse all matching dishes"):
            view = menu[["display", "serving_g", "calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g"]].rename(columns={"display":"Dish", "serving_g":"Serving (g)", "calories_kcal":"Calories (kcal)", "protein_g":"Protein (g)", "carbs_g":"Carbs (g)", "fat_g":"Fat (g)", "fiber_g":"Fibre (g)"})
            st.dataframe(view, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3 — PROJECT STORY / TRANSPARENCY
# ============================================================
with about_tab:
    st.markdown('<div class="eyebrow">03 / BEHIND THE TADKA</div><div class="section-title">A little AI in the Indian kitchen.</div><div class="section-sub">Masala Metrics brings image classification and a reference nutrition database together in one simple food explorer.</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><div style="font-size:30px">🧠</div><div class="section-title" style="font-size:25px">The food recogniser</div><p style="color:#776f68;font-size:13px;line-height:1.8">A trained EfficientNetB0 image-classification model predicts a food category from the uploaded photo. The app displays the leading match and the five highest model scores.</p><b style="color:#a84b32;font-size:12px">TENSORFLOW · KERAS · EFFICIENTNETB0</b></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div style="font-size:30px">📚</div><div class="section-title" style="font-size:25px">The nutrition lookup</div><p style="color:#776f68;font-size:13px;line-height:1.8">After identifying a dish, the app finds its matching row in the nutrition CSV and scales the reference values to the serving size selected by the user.</p><b style="color:#a84b32;font-size:12px">PYTHON · PANDAS · STREAMLIT</b></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="note"><b>Honest plate, honest numbers.</b><br>The classifier predicts a dish category; it does not estimate portion weight. Nutrition values are reference estimates and may differ from the food on your plate. Model scores should not be interpreted as guaranteed probabilities. This tool is for general information, not medical or dietary advice.</div>', unsafe_allow_html=True)
    st.markdown("**What’s in this project**")
    st.write("- EfficientNetB0 model for 96 food categories\n- JSON class-label mapping\n- CSV nutrition reference database\n- Streamlit interface with image upload, prediction ranking and serving-size scaling")

st.markdown('<div class="footer"><b>MASALA METRICS</b> &nbsp;✦&nbsp; Made with a little desi tadka.<br>Eat with curiosity. Explore with care. Nutrition values are approximate and are not medical advice.</div>', unsafe_allow_html=True)
