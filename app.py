# ============================================================
# OUTFIT STYLE FINDER
# Streamlit Deployment — GMM Clustering Model
# ============================================================
# Files required in the same folder as this app.py:
#   gmm.pkl
#   scaler.pkl
#   pca.pkl
#   tfidf.pkl
#   feature_info.json
#   cluster_results.json
# ============================================================

import io
import json
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Outfit Style Finder",
    page_icon="👗",
    layout="wide"
)

# ============================================================
# LOAD MODELS — cached so they only load once per session
# ============================================================

@st.cache_resource
def load_models():
    gmm     = joblib.load("gmm.pkl")
    scaler  = joblib.load("scaler.pkl")
    pca     = joblib.load("pca.pkl")
    tfidf   = joblib.load("tfidf.pkl")
    with open("feature_info.json") as f:
        info = json.load(f)
    with open("cluster_results.json") as f:
        cluster_results = json.load(f)
    return gmm, scaler, pca, tfidf, info, cluster_results

gmm, scaler, pca, tfidf, info, cluster_results = load_models()

N_CLUSTERS = info["n_clusters"]
IMAGE_SIZE = tuple(info["image_size"])

# ============================================================
# CLUSTER LABELS
# Descriptive names for each cluster.
# Update these after inspecting your cluster_outfit_samples.png
# to give each cluster a meaningful style name.
# ============================================================

CLUSTER_LABELS = {
    0: ("🌸 Soft & Feminine",    "Outfits with soft colors, floral patterns, and delicate accessories."),
    1: ("🖤 Bold & Edgy",        "Dark tones, structured pieces, and statement accessories."),
    2: ("☀️ Casual & Relaxed",   "Comfortable everyday wear with neutral or warm color palettes."),
}

# Fallback for N_CLUSTERS > 3
for i in range(3, N_CLUSTERS):
    CLUSTER_LABELS[i] = (f"✨ Style Group {i}", f"Cluster {i} outfits.")

# ============================================================
# FEATURE EXTRACTION — must match training exactly
# ============================================================

def extract_color_features(img_rgb: np.ndarray) -> np.ndarray:
    hist_r = np.histogram(img_rgb[:, :, 0], bins=8, range=(0, 256))[0]
    hist_g = np.histogram(img_rgb[:, :, 1], bins=8, range=(0, 256))[0]
    hist_b = np.histogram(img_rgb[:, :, 2], bins=8, range=(0, 256))[0]
    hist   = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    return hist / (hist.sum() + 1e-7)

def extract_hsv_features(img_rgb: np.ndarray) -> np.ndarray:
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist_h  = np.histogram(img_hsv[:, :, 0], bins=18, range=(0, 180))[0]
    hist_s  = np.histogram(img_hsv[:, :, 1], bins=8,  range=(0, 256))[0]
    hist    = np.concatenate([hist_h, hist_s]).astype(float)
    return hist / (hist.sum() + 1e-7)

def extract_texture_features(img_gray: np.ndarray) -> np.ndarray:
    lbp  = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    hist = np.histogram(lbp, bins=26, range=(0, 26))[0].astype(float)
    return hist / (hist.sum() + 1e-7)

def extract_visual_features(pil_image: Image.Image) -> np.ndarray:
    img_rgb  = np.array(pil_image.resize(IMAGE_SIZE))
    img_gray = rgb2gray(img_rgb)
    color    = extract_color_features(img_rgb)
    hsv      = extract_hsv_features(img_rgb)
    texture  = extract_texture_features(img_gray)
    return np.concatenate([color, hsv, texture])   # 76-D

def predict_cluster(images: list) -> tuple:
    """
    Given a list of PIL images (one per clothing item),
    extract features, mean pool, combine with blank TF-IDF,
    scale, reduce with PCA, and predict cluster with GMM.
    Returns (cluster_id, confidence_scores_array).
    """
    # Extract visual features per item
    item_feats = []
    for img in images:
        feat = extract_visual_features(img)
        item_feats.append(feat)

    # Mean pool all item features → one outfit vector
    visual_vec = np.mean(item_feats, axis=0).reshape(1, -1)

    # TF-IDF: no item IDs available from user upload,
    # so we pass a blank document — TF-IDF contributes zeros.
    # The visual features carry the prediction.
    tfidf_vec = tfidf.transform([""]).toarray()

    # Combine visual + TF-IDF
    combined = np.concatenate([visual_vec, tfidf_vec], axis=1)

    # Scale → PCA → GMM predict
    scaled    = scaler.transform(combined)
    reduced   = pca.transform(scaled)
    cluster   = int(gmm.predict(reduced)[0])
    probs     = gmm.predict_proba(reduced)[0]

    return cluster, probs

# ============================================================
# SIMILAR OUTFITS FROM CLUSTER RESULTS
# ============================================================

def get_similar_outfit_ids(cluster_id: int, top_n: int = 5) -> list:
    """
    Returns set_ids of outfits assigned to the predicted cluster
    from the training data — used to show similar style examples.
    """
    matching = [
        r["set_id"] for r in cluster_results
        if r["gmm"] == cluster_id
    ]
    # Return a random sample so results vary each time
    rng = np.random.default_rng()
    sample_size = min(top_n, len(matching))
    return list(rng.choice(matching, size=sample_size, replace=False))

# ============================================================
# UI — HEADER
# ============================================================

st.title("👗 Outfit Style Finder")
st.markdown(
    "Upload your clothing items and discover which **style group** "
    "your outfit belongs to. Powered by GMM clustering trained on "
    "the Polyvore fashion dataset."
)
st.divider()

# ============================================================
# UI — FILE UPLOAD
# ============================================================

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("📤 Upload Your Outfit Items")
    st.caption(
        "Upload 1–6 images of individual clothing or accessory items "
        "(top, pants, shoes, bag, etc.). Each image = one item."
    )
    uploaded_files = st.file_uploader(
        "Choose item images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload up to 6 clothing item images."
    )

with col_info:
    st.subheader("ℹ️ How It Works")
    st.markdown("""
    1. Upload your clothing item images
    2. The app extracts **color**, **HSV**, and **texture** features
    3. Features are scaled and reduced with **PCA**
    4. **GMM** predicts your outfit's style cluster
    5. Similar outfits from the same cluster are shown
    """)

# ============================================================
# UI — PREVIEW UPLOADED ITEMS
# ============================================================

if uploaded_files:
    if len(uploaded_files) > 6:
        st.warning("Please upload a maximum of 6 items. Only the first 6 will be used.")
        uploaded_files = uploaded_files[:6]

    st.divider()
    st.subheader("🧺 Your Outfit Items")

    # Show uploaded items in a row
    cols = st.columns(len(uploaded_files))
    pil_images = []

    for i, (col, f) in enumerate(zip(cols, uploaded_files)):
        img = Image.open(f).convert("RGB")
        pil_images.append(img)
        with col:
            st.image(img, caption=f"Item {i + 1}", use_column_width=True)

    st.divider()

    # ============================================================
    # PREDICTION
    # ============================================================

    with st.spinner("Analyzing your outfit..."):
        cluster_id, probs = predict_cluster(pil_images)

    label, description = CLUSTER_LABELS.get(
        cluster_id, (f"Style Group {cluster_id}", "")
    )
    confidence = float(probs[cluster_id]) * 100

    # ============================================================
    # UI — RESULT
    # ============================================================

    st.subheader("🎯 Your Style Prediction")

    res_col, conf_col = st.columns([2, 1])

    with res_col:
        st.markdown(f"## {label}")
        st.markdown(f"*{description}*")

    with conf_col:
        st.metric(
            label="GMM Confidence",
            value=f"{confidence:.1f}%",
            help="How strongly GMM assigns your outfit to this cluster."
        )

    # Probability bar chart for all clusters
    st.markdown("**Cluster probability breakdown:**")
    prob_cols = st.columns(N_CLUSTERS)
    for cid, (pcol, prob) in enumerate(zip(prob_cols, probs)):
        lbl, _ = CLUSTER_LABELS.get(cid, (f"Cluster {cid}", ""))
        with pcol:
            st.metric(label=lbl, value=f"{prob * 100:.1f}%")
            st.progress(float(prob))

    st.divider()

    # ============================================================
    # UI — SIMILAR OUTFITS FROM TRAINING DATA
    # ============================================================

    st.subheader("👀 Similar Outfits from This Style Group")
    st.caption(
        f"These are real Polyvore outfits that were clustered into "
        f"**{label}** during training. "
        f"Set IDs are shown for reference."
    )

    similar_ids = get_similar_outfit_ids(cluster_id, top_n=5)

    if similar_ids:
        id_cols = st.columns(len(similar_ids))
        for col, sid in zip(id_cols, similar_ids):
            with col:
                st.markdown(
                    f"<div style='text-align:center; padding:12px; "
                    f"background:#f0f0f0; border-radius:8px; "
                    f"font-size:12px; color:#333;'>"
                    f"📦 Outfit<br><b>{sid}</b></div>",
                    unsafe_allow_html=True
                )
        st.caption(
            "💡 To see the actual outfit images, look up these set IDs "
            "in your cluster_outfit_samples.png visualization."
        )
    else:
        st.info("No similar outfits found for this cluster.")

    st.divider()

    # ============================================================
    # UI — ALL CLUSTER OVERVIEW
    # ============================================================

    with st.expander("📊 View All Style Clusters"):
        for cid in range(N_CLUSTERS):
            lbl, desc = CLUSTER_LABELS.get(cid, (f"Cluster {cid}", ""))
            count = sum(1 for r in cluster_results if r["gmm"] == cid)
            highlight = "🔵 " if cid == cluster_id else ""
            st.markdown(
                f"**{highlight}{lbl}** — {count:,} outfits in training data  \n"
                f"*{desc}*"
            )

else:
    # ---- Empty state ----
    st.info("👆 Upload your outfit item images above to get started.")

    st.divider()
    st.subheader("📊 Style Clusters Overview")
    st.caption("These are the style groups learned from 5,000 Polyvore outfits.")

    for cid in range(N_CLUSTERS):
        lbl, desc = CLUSTER_LABELS.get(cid, (f"Cluster {cid}", ""))
        count = sum(1 for r in cluster_results if r["gmm"] == cid)
        st.markdown(f"**{lbl}** — {count:,} outfits  \n*{desc}*")
        st.divider()

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px; margin-top:40px;'>"
    "Outfit Style Finder · GMM Clustering · Polyvore Dataset · PTF30 Final Project"
    "</div>",
    unsafe_allow_html=True
)
