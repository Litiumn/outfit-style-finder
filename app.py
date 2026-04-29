import json
import random
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import os
import pickle

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FitBoard",
    page_icon="✦",
    layout="wide"
)

# ============================================================
# 🎨 CUSTOM CSS — Luxury Editorial Dark Mode
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ── Root & Body ── */
:root {
    --bg:       #0c0c0e;
    --surface:  #141416;
    --surface2: #1c1c1f;
    --border:   rgba(255,255,255,0.07);
    --accent:   #e8c97a;
    --accent2:  #c9a84c;
    --text:     #f0ede8;
    --muted:    #888680;
    --tag-bg:   rgba(232,201,122,0.10);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

/* ── Hero header ── */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 56px 0 32px;
    position: relative;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%; transform: translateX(-50%);
    width: 560px; height: 280px;
    background: radial-gradient(ellipse at center,
        rgba(232,201,122,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.logo-lockup {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
}
.logo-icon {
    width: 54px; height: 54px;
    background: linear-gradient(135deg, #e8c97a 0%, #b8882e 100%);
    border-radius: 15px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 8px 28px rgba(232,201,122,0.40),
                inset 0 1px 0 rgba(255,255,255,0.2);
}
.logo-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 700;
    letter-spacing: -1.5px;
    color: var(--text);
    line-height: 1;
}
.logo-wordmark em {
    color: var(--accent);
    font-style: italic;
}

.hero-tagline {
    font-size: 11px;
    font-weight: 300;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--muted);
}
.hero-divider {
    width: 40px; height: 1px;
    background: linear-gradient(90deg,
        transparent, var(--accent), transparent);
    margin: 20px auto 0;
}

/* ── Section labels ── */
.section-label {
    font-size: 10px;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: var(--accent);
    margin: 40px 0 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Upload zone ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed rgba(232,201,122,0.22) !important;
    border-radius: 18px !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(232,201,122,0.5) !important;
    background: var(--surface2) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--muted) !important;
}

/* ── Outfit preview images ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--border);
    transition: transform 0.3s;
    width: 100%;
}
[data-testid="stImage"] img:hover { transform: scale(1.02); }

/* ── Cluster badge ── */
.cluster-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--tag-bg);
    border: 1px solid rgba(232,201,122,0.18);
    color: var(--accent);
    font-size: 10.5px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin: 14px 0 30px;
}

/* ── Outfit card ── */
.outfit-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-bottom: 20px;
    overflow: hidden;
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
    animation: fadeUp 0.4s ease both;
}
.outfit-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 24px 56px rgba(0,0,0,0.5);
    border-color: rgba(232,201,122,0.2);
}

.outfit-card-header {
    padding: 16px 18px 12px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
}
.outfit-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-style: italic;
    color: var(--text);
    line-height: 1.35;
    margin: 0;
}
.outfit-card-num {
    font-size: 9.5px;
    letter-spacing: 1px;
    color: var(--muted);
    background: var(--surface2);
    border-radius: 6px;
    padding: 3px 8px;
    white-space: nowrap;
    margin-left: 8px;
    margin-top: 2px;
    flex-shrink: 0;
}

/* ── Horizontal items strip ── */
.items-strip {
    display: flex;
    flex-direction: row;
    gap: 10px;
    padding: 14px 16px 16px;
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: #2a2a2a transparent;
}
.items-strip::-webkit-scrollbar { height: 3px; }
.items-strip::-webkit-scrollbar-thumb {
    background: #2a2a2a; border-radius: 2px;
}

.item-tile {
    flex: 0 0 auto;
    width: 88px;
    display: flex;
    flex-direction: column;
    gap: 7px;
    align-items: center;
}
.item-tile img {
    width: 88px !important;
    height: 108px !important;
    object-fit: cover;
    border-radius: 10px;
    border: 1px solid var(--border);
    display: block;
    transition: transform 0.2s;
}
.item-tile img:hover { transform: scale(1.04); }
.item-tag {
    font-size: 9px;
    letter-spacing: 0.4px;
    color: var(--muted);
    text-align: center;
    line-height: 1.3;
    text-transform: capitalize;
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Refresh button ── */
.stButton > button {
    background: linear-gradient(135deg, #e8c97a 0%, #b8882e 100%) !important;
    color: #0c0c0e !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 10px 32px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(232,201,122,0.35) !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
}

/* ── Streamlit info/warning overrides ── */
[data-testid="stNotification"],
.stAlert {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--muted) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 10px;
    letter-spacing: 2.5px;
    color: #2a2a2a;
    text-transform: uppercase;
    padding: 48px 0 28px;
}

/* ── Fade-up entrance ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_color_features(img_rgb):
    hist_r = np.histogram(img_rgb[:, :, 0], bins=8, range=(0, 256))[0]
    hist_g = np.histogram(img_rgb[:, :, 1], bins=8, range=(0, 256))[0]
    hist_b = np.histogram(img_rgb[:, :, 2], bins=8, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    return hist / (hist.sum() + 1e-7)

def extract_hsv_features(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist_h  = np.histogram(img_hsv[:, :, 0], bins=18, range=(0, 180))[0]
    hist_s  = np.histogram(img_hsv[:, :, 1], bins=8, range=(0, 256))[0]
    hist = np.concatenate([hist_h, hist_s]).astype(float)
    return hist / (hist.sum() + 1e-7)

def extract_texture_features(img_gray):
    lbp  = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    hist = np.histogram(lbp, bins=26, range=(0, 26))[0].astype(float)
    return hist / (hist.sum() + 1e-7)

def extract_visual_features(pil_image, IMAGE_SIZE):
    img_rgb  = np.array(pil_image.resize(IMAGE_SIZE))
    img_gray = rgb2gray(img_rgb)
    color    = extract_color_features(img_rgb)
    hsv      = extract_hsv_features(img_rgb)
    texture  = extract_texture_features(img_gray)
    return np.concatenate([color, hsv, texture])

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():
    gmm     = joblib.load("gmm.pkl")
    scaler  = joblib.load("scaler.pkl")
    pca     = joblib.load("pca.pkl")
    tfidf   = joblib.load("tfidf.pkl")

    with open("feature_info.json") as f:
        info = json.load(f)

    with open("final_merged.json") as f:
        merged_data = json.load(f)

    IMAGE_SIZE = tuple(info["image_size"])

    VECTOR_CACHE = "outfit_vectors.pkl"

    if os.path.exists(VECTOR_CACHE):
        with open(VECTOR_CACHE, "rb") as f:
            outfit_vectors = pickle.load(f)
    else:
        st.write("⚙️ Building outfit vectors...")

        outfit_vectors = {}

        for outfit in merged_data:
            feats = []

            for item in outfit["items"]:
                item_id = item["item_id"]
                img_path = f"images/{item_id}.jpg"

                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        feat = extract_visual_features(img, IMAGE_SIZE)
                        feats.append(feat)
                    except:
                        continue

            if feats:
                visual_vec = np.mean(feats, axis=0).reshape(1, -1)
                tfidf_vec = tfidf.transform([""]).toarray()

                combined = np.concatenate([visual_vec, tfidf_vec], axis=1)
                scaled = scaler.transform(combined)
                reduced = pca.transform(scaled)

                outfit_vectors[outfit["set_id"]] = reduced[0]

        with open(VECTOR_CACHE, "wb") as f:
            pickle.dump(outfit_vectors, f)

    return gmm, scaler, pca, tfidf, IMAGE_SIZE, merged_data, outfit_vectors


gmm, scaler, pca, tfidf, IMAGE_SIZE, merged_data, outfit_vectors = load_data()

# ============================================================
# SESSION STATE
# ============================================================

if "shown_outfits" not in st.session_state:
    st.session_state.shown_outfits = set()

# ============================================================
# MODEL
# ============================================================

def predict_cluster_and_vector(images):
    feats = [extract_visual_features(img, IMAGE_SIZE) for img in images]
    visual_vec = np.mean(feats, axis=0).reshape(1, -1)

    tfidf_vec = tfidf.transform([""]).toarray()
    combined = np.concatenate([visual_vec, tfidf_vec], axis=1)

    scaled = scaler.transform(combined)
    reduced = pca.transform(scaled)

    cluster = int(gmm.predict(reduced)[0])
    return cluster, reduced[0]

# ============================================================
# RECOMMENDER (HYBRID)
# ============================================================

def get_similar_outfits(cluster_id, user_vector, shown_ids, top_n=10):

    candidates = [
        o for o in merged_data
        if o["gmm_cluster"] == cluster_id
        and o["set_id"] in outfit_vectors
    ]

    if not candidates:
        candidates = [o for o in merged_data if o["gmm_cluster"] == cluster_id]

    if not candidates:
        candidates = merged_data

    unseen = [o for o in candidates if o["set_id"] not in shown_ids]

    if not unseen:
        st.session_state.shown_outfits.clear()
        unseen = candidates

    ranked = []

    for outfit in unseen:
        sid = outfit["set_id"]
        if sid in outfit_vectors:
            dist = np.linalg.norm(user_vector - outfit_vectors[sid])
            ranked.append((dist, outfit))

    if ranked:
        ranked.sort(key=lambda x: x[0])
        return [o for _, o in ranked[:top_n]]

    return random.sample(unseen, min(top_n, len(unseen)))

# ============================================================
# UI HEADER
# ============================================================

st.markdown("""
<div class="hero-wrap">
    <div class="logo-lockup">
        <div class="logo-icon">✦</div>
        <div class="logo-wordmark">Fit<em>Board</em></div>
    </div>
    <div class="hero-tagline">Discover outfits that match your vibe</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UPLOAD
# ============================================================

st.markdown('<div class="section-label">✦ &nbsp; Upload Your Pieces</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop your outfit images here — up to 6 pieces",
    type=["jpg", "png", "jpeg", "webp"],
    accept_multiple_files=True,
    label_visibility="visible"
)

# ============================================================
# MAIN
# ============================================================

# ============================================================
# MAIN
# ============================================================

if uploaded_files:
    uploaded_files = uploaded_files[:6]

    st.markdown('<div class="section-label">✦ &nbsp; Your Outfit</div>', unsafe_allow_html=True)

    cols = st.columns(len(uploaded_files))
    pil_images = []

    for col, f in zip(cols, uploaded_files):
        img = Image.open(f).convert("RGB")
        pil_images.append(img)
        col.image(img, use_container_width=True)

    cluster_id, user_vector = predict_cluster_and_vector(pil_images)

    st.markdown(
        f'<div style="text-align:center">'
        f'<span class="cluster-badge">✦ &nbsp; Style Profile · Cluster {cluster_id}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-label">✦ &nbsp; Mood Board</div>', unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 5])
    with btn_col:
        if st.button("↻  Refresh"):
            st.rerun()

    outfits = get_similar_outfits(
        cluster_id,
        user_vector,
        st.session_state.shown_outfits
    )

    if not outfits:
        st.warning("No outfits available right now.")
    else:
        for outfit in outfits:
            st.session_state.shown_outfits.add(outfit["set_id"])

        grid_cols = st.columns(4)

        for i, outfit in enumerate(outfits):
            col = grid_cols[i % 4]
            with col:
                title = outfit["outfit"]["title"] if outfit["outfit"] else "Untitled"
                item_count = len(outfit["items"])

                # ── Card header ──
                header_html = (
                    f'<div class="outfit-card">'
                    f'  <div class="outfit-card-header">'
                    f'    <div class="outfit-card-title">{title}</div>'
                    f'    <div class="outfit-card-num">{item_count} pieces</div>'
                    f'  </div>'
                    f'  <div class="items-strip" id="strip-{i}">'
                )

                items_html = ""
                for item in sorted(outfit["items"], key=lambda x: x["index"]):
                    item_id   = item["item_id"]
                    metadata  = item["metadata"]
                    img_path  = f"images/{item_id}.jpg"
                    cat_label = (metadata.get("semantic_category", "") if metadata else "").strip()

                    if os.path.exists(img_path):
                        # Encode image as base64 for inline rendering
                        import base64
                        with open(img_path, "rb") as img_f:
                            b64 = base64.b64encode(img_f.read()).decode()
                        items_html += (
                            f'<div class="item-tile">'
                            f'  <img src="data:image/jpeg;base64,{b64}" />'
                            f'  <div class="item-tag">{cat_label}</div>'
                            f'</div>'
                        )

                footer_html = '</div></div>'  # close items-strip + outfit-card

                st.markdown(header_html + items_html + footer_html, unsafe_allow_html=True)

    st.markdown('<div class="footer">FitBoard &nbsp;·&nbsp; Style Intelligence</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 64px 0 48px;
        color: #444;
        font-size: 13px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    ">
        Upload outfit images above to generate your FitBoard ✦
    </div>
    """, unsafe_allow_html=True)