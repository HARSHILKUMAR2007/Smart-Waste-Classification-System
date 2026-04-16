"""
Smart Waste Classification System — Streamlit Web App
Classifies garbage images into 12 categories using a trained CNN (MobileNetV2).
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:   #0d1117;
    --bg-card:      #161b22;
    --bg-elevated:  #21262d;
    --accent-green: #3fb950;
    --accent-blue:  #58a6ff;
    --accent-orange:#ffa657;
    --accent-red:   #ff7b72;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       #30363d;
    --radius:       12px;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main Area ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0d2818 0%, #0d1117 40%, #0a1f35 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(63,185,80,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3fb950, #58a6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: var(--accent-green);
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 1rem;
}

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, rgba(63,185,80,0.1), rgba(88,166,255,0.05));
    border: 1px solid rgba(63,185,80,0.35);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-green);
    margin: 0.3rem 0;
    text-transform: capitalize;
}
.result-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: var(--accent-blue);
}

/* ── Confidence Bar ── */
.conf-bar-wrap {
    background: var(--bg-elevated);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0 1rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #3fb950, #58a6ff);
    transition: width 0.8s ease;
}

/* ── Eco Tip ── */
.eco-tip {
    background: linear-gradient(135deg, rgba(255,166,87,0.08), rgba(255,166,87,0.03));
    border-left: 3px solid var(--accent-orange);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.3rem;
    margin: 1rem 0;
}
.eco-tip-title {
    color: var(--accent-orange);
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.eco-tip-body { color: var(--text-primary); font-size: 0.93rem; line-height: 1.6; }

/* ── Warning / Error ── */
.warn-box {
    background: rgba(255,123,114,0.08);
    border: 1px solid rgba(255,123,114,0.3);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: var(--accent-red);
    font-size: 0.9rem;
}

/* ── Class chip in sidebar ── */
.class-chip {
    display: inline-block;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 0.2rem;
}

/* ── Streamlit upload button override ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-elevated) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Top prediction rows ── */
.top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
}
.top-row:last-child { border-bottom: none; }
.top-row-label { text-transform: capitalize; color: var(--text-primary); }
.top-row-pct {
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-blue);
    font-weight: 600;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & ECO TIPS
# ─────────────────────────────────────────────────────────────
MODEL_PATH         = os.path.join("model", "model.h5")
CLASS_INDICES_PATH = os.path.join("model", "class_indices.json")
IMG_SIZE           = 224

ECO_TIPS = {
    "battery": {
        "icon": "🔋",
        "tip": "Batteries contain toxic chemicals like lead and mercury. Never throw them in general waste. Drop them off at dedicated battery recycling points in electronics stores or municipal collection centres."
    },
    "biological": {
        "icon": "🌱",
        "tip": "Biological/organic waste can be composted to create rich fertiliser. Start a home compost bin or use your local organic waste collection. Composting reduces methane emissions from landfills."
    },
    "brown-glass": {
        "icon": "🍺",
        "tip": "Brown glass is 100% recyclable endlessly without quality loss. Rinse the bottle, remove the cap, and place it in the glass recycling bin. Avoid mixing with ceramics or Pyrex."
    },
    "cardboard": {
        "icon": "📦",
        "tip": "Flatten cardboard boxes to save space in recycling bins. Keep it dry — wet cardboard loses its fibre quality and may not be accepted. Remove any plastic tape or foam inserts."
    },
    "clothes": {
        "icon": "👕",
        "tip": "Donate wearable clothes to charity or clothing banks. For worn-out items, look for textile recycling bins (H&M, Uniqlo, etc. run take-back programs). Avoid sending textiles to landfill."
    },
    "green-glass": {
        "icon": "🍾",
        "tip": "Green glass goes to the green/mixed glass recycling container. Glass can be recycled indefinitely. Rinsing bottles removes odours and contamination that could spoil recycling batches."
    },
    "metal": {
        "icon": "🥫",
        "tip": "Metal cans are highly valuable recyclables — aluminium recycling uses 95% less energy than new production. Rinse cans and crush them to save space. Scrap metal dealers also accept larger metal items."
    },
    "paper": {
        "icon": "📄",
        "tip": "Paper is easily recyclable but must be clean and dry. Shredded paper, newspapers, and office paper are all accepted. Avoid recycling wax-coated paper, tissue, or food-soiled paper."
    },
    "plastic": {
        "icon": "🧴",
        "tip": "Check the resin code (1–7) on the bottom. Types 1 (PET) and 2 (HDPE) are most widely accepted. Rinse containers, remove lids, and avoid small plastic pieces that jam sorting machinery."
    },
    "shoes": {
        "icon": "👟",
        "tip": "Donate usable shoes to charity shops or shoe banks. Brands like Nike (Reuse-a-Shoe) and Adidas run recycling programs for worn-out footwear. Avoid sending shoes to landfill — soles take centuries to decompose."
    },
    "trash": {
        "icon": "🗑️",
        "tip": "General waste that can't be recycled should be minimised. Before binning, double-check whether any part can be composted or recycled. Reducing consumption and buying products with less packaging is the best long-term solution."
    },
    "white-glass": {
        "icon": "🥛",
        "tip": "Clear/white glass is the most versatile for recycling and is used to make new glass containers. Place in the clear glass section of your local bottle bank. Remove corks and plastic sleeves."
    },
}

DEFAULT_CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]


# ─────────────────────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    """Load the Keras model and class index mapping."""
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        return None, None, "Model file not found. Please train the model first (run train.py)."

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    # Load class indices
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        # Invert to {index: class_name}
        idx_to_class = {v: k for k, v in class_indices.items()}
    else:
        idx_to_class = {i: name for i, name in enumerate(DEFAULT_CLASSES)}

    return model, idx_to_class, None


# ─────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize, normalise and batch-expand an image for model input."""
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def predict(model, img_array: np.ndarray, idx_to_class: dict):
    """Run inference and return sorted (class, confidence) list."""
    preds = model.predict(img_array, verbose=0)[0]          # (num_classes,)
    results = [(idx_to_class[i], float(preds[i])) for i in range(len(preds))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results                                           # [(label, conf), ...]


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ♻️ About")
        st.markdown("""
        <div style="color:#8b949e;font-size:0.88rem;line-height:1.7">
        This system uses a <strong style="color:#e6edf3">MobileNetV2</strong> CNN
        trained on ~15,000 garbage images across <strong style="color:#e6edf3">12 waste categories</strong>.
        Upload any photo and get an instant classification with eco-friendly advice.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Supported Classes**")
        chips_html = "".join(
            f'<span class="class-chip">{ECO_TIPS[c]["icon"]} {c}</span>'
            for c in DEFAULT_CLASSES
        )
        st.markdown(f'<div style="line-height:2.2">{chips_html}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Model Info**")
        st.markdown("""
        <div style="color:#8b949e;font-size:0.82rem;line-height:1.8">
        • Architecture: MobileNetV2 + custom head<br>
        • Input size: 224 × 224 px<br>
        • Training: Transfer Learning + Fine-tuning<br>
        • Dataset: Kaggle Garbage Classification<br>
        • Classes: 12
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Hero ───────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">AI-Powered · CNN · MobileNetV2</div>
        <h1 class="hero-title">♻️ Smart Waste Classification System</h1>
        <p class="hero-subtitle">
            Upload a photo of any waste item and our AI will instantly classify it
            and guide you on the eco-friendly way to dispose of it.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Model ─────────────────────────────
    with st.spinner("Loading model..."):
        model, idx_to_class, error = load_model_and_classes()

    if error:
        st.markdown(f'<div class="warn-box">⚠️ {error}</div>', unsafe_allow_html=True)
        st.info("**Quick start:** Run `python train.py` to train the model, then restart this app.")
        return

    # ── Layout: two columns ────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="card-title">📤 Upload Waste Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPG, PNG, WEBP, BMP"
        )

        if uploaded:
            # Validate
            try:
                img = Image.open(uploaded)
                img.verify()                        # Check file integrity
                img = Image.open(uploaded)          # Reopen after verify
            except Exception:
                st.markdown('<div class="warn-box">⚠️ Invalid or corrupted image file. Please upload a valid image.</div>',
                            unsafe_allow_html=True)
                return

            st.image(img, caption="Uploaded image", use_container_width=True)
            st.markdown(f"""
            <div style="color:var(--text-muted);font-size:0.8rem;margin-top:0.5rem">
            📐 {img.width} × {img.height} px &nbsp;|&nbsp; 🎨 {img.mode}
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        if not uploaded:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 2rem">
                <div style="font-size:4rem;margin-bottom:1rem">🔍</div>
                <div style="color:var(--text-muted);font-size:0.95rem">
                    Upload an image on the left to see the classification results here.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-title">🎯 Classification Result</div>', unsafe_allow_html=True)

            with st.spinner("Analysing image..."):
                try:
                    img_array = preprocess_image(img)
                    results   = predict(model, img_array, idx_to_class)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠️ Prediction error: {e}</div>',
                                unsafe_allow_html=True)
                    return

            top_label, top_conf = results[0]
            tip_data = ECO_TIPS.get(top_label, {"icon": "♻️", "tip": "Please dispose of this item responsibly."})

            # ── Result box ──
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:3rem">{tip_data['icon']}</div>
                <div class="result-label">{top_label.replace('-', ' ')}</div>
                <div class="result-confidence">Confidence: {top_conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown(f"""
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{top_conf*100:.1f}%"></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Eco tip ──
            st.markdown(f"""
            <div class="eco-tip">
                <div class="eco-tip-title">🌍 Eco-Friendly Disposal Tip</div>
                <div class="eco-tip-body">{tip_data['tip']}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Top 5 predictions ──
            st.markdown('<div class="card-title" style="margin-top:1.2rem">Top 5 Predictions</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="card" style="padding:1rem 1.2rem">', unsafe_allow_html=True)
            for label, conf in results[:5]:
                bar_width = int(conf * 100)
                icon = ECO_TIPS.get(label, {}).get("icon", "♻️")
                st.markdown(f"""
                <div class="top-row">
                    <span class="top-row-label">{icon} {label.replace('-', ' ')}</span>
                    <span class="top-row-pct">{conf*100:.1f}%</span>
                </div>
                <div style="background:#21262d;border-radius:4px;height:4px;margin-bottom:4px">
                    <div style="width:{bar_width}%;height:4px;border-radius:4px;
                                background:linear-gradient(90deg,#3fb950,#58a6ff)"></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────
    st.markdown("""
    <hr style="border-color:#30363d;margin:2rem 0 1rem">
    <div style="text-align:center;color:#8b949e;font-size:0.8rem">
        Built with TensorFlow · MobileNetV2 · Streamlit &nbsp;|&nbsp;
        Dataset: <a href="https://www.kaggle.com/datasets/mostafaabla/garbage-classification"
                    target="_blank" style="color:#58a6ff;text-decoration:none">
            Kaggle Garbage Classification
        </a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()