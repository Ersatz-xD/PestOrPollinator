import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="PestOrPollinator AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Fraunces:ital,wght@1,400;1,500&display=swap');

    :root {
        color-scheme: dark;
        --bg-0: #090F08;
        --bg-1: #10190D;
        --card: #141F13;
        --card-border: rgba(163, 255, 107, 0.14);
        --text-primary: #EAF2E4;
        --text-muted: #90A88A;
        --accent: #A3FF6B;
        --accent-dim: rgba(163, 255, 107, 0.12);
        --pest: #FF6B5B;
        --pollinator: #A3FF6B;
        --neutral: #6BC7FF;
    }

    html, body, [class*="css"], [data-testid] {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Outlined' !important;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #14231080 0%, transparent 45%),
                    radial-gradient(circle at 90% 10%, #0f2a1580 0%, transparent 40%),
                    var(--bg-0) !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] div {
        color: var(--text-primary);
    }

    [data-testid="stCaptionContainer"], .stCaption, small {
        color: var(--text-muted) !important;
    }

    .accent-script {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 500;
        color: var(--accent) !important;
    }

    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .nature-card {
        background-color: var(--card);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0px 12px 32px rgba(0, 0, 0, 0.35);
        border: 1px solid var(--card-border);
        margin-bottom: 20px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0F1A0C 0%, #1C2E16 100%);
        border: 1px solid var(--card-border);
        border-radius: 22px;
        padding: 36px;
        color: var(--text-primary) !important;
        box-shadow: 0px 20px 48px rgba(0, 0, 0, 0.45);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, var(--accent-dim) 0%, transparent 70%);
    }

    .badge-pollinator, .badge-pest, .badge-neutral {
        font-family: 'JetBrains Mono', monospace;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-block;
        border: 1px solid;
    }

    .badge-pollinator {
        background-color: rgba(163, 255, 107, 0.1);
        color: var(--pollinator) !important;
        border-color: rgba(163, 255, 107, 0.35);
    }

    .badge-pest {
        background-color: rgba(255, 107, 91, 0.1);
        color: var(--pest) !important;
        border-color: rgba(255, 107, 91, 0.35);
    }

    .badge-neutral {
        background-color: rgba(107, 199, 255, 0.1);
        color: var(--neutral) !important;
        border-color: rgba(107, 199, 255, 0.35);
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--accent) !important;
        margin: 0;
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-1) !important;
        border-right: 1px solid var(--card-border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
        color: var(--text-muted) !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--card-border);
    }

    div[data-testid="stFileUploader"] {
        background-color: var(--card);
        border-radius: 16px;
        padding: 20px 24px;
        border: 1.5px dashed rgba(163, 255, 107, 0.4);
    }

    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        flex-wrap: wrap;
        gap: 16px;
        border: none !important;
        padding: 0 !important;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        align-items: center !important;
        gap: 12px;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text-primary) !important;
        opacity: 1 !important;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text-muted) !important;
    }

    div[data-testid="stFileUploader"] button {
        position: relative;
        background-color: var(--accent) !important;
        color: var(--bg-0) !important;
        border-radius: 50px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        border: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px;
        white-space: nowrap;
        overflow: hidden;
        flex-shrink: 0;
    }

    div[data-testid="stFileUploader"] button * {
        color: var(--bg-0) !important;
        -webkit-text-fill-color: var(--bg-0) !important;
    }

    div[data-testid="stFileUploader"] button [data-testid="stIconMaterial"] {
        display: none !important;
    }

    div[role="radiogroup"] label {
        color: var(--text-primary) !important;
    }

    div[role="radiogroup"] label span {
        color: var(--text-primary) !important;
    }

    .stButton>button {
        background-color: var(--accent) !important;
        color: var(--bg-0) !important;
        border-radius: 50px !important;
        padding: 12px 28px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0px 8px 20px rgba(163, 255, 107, 0.2) !important;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: var(--card) !important;
    }

    hr {
        border-color: var(--card-border) !important;
    }

    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-0);
    }

    ::-webkit-scrollbar-thumb {
        background: #2A3E25;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PATHS = {
    "embeddings": "embeddings_and_labels/embeddings.npy",
    "labels": "embeddings_and_labels/labels.npy",
    "umap": "embeddings_and_labels/umap_df.csv",
    "model_2048": "models/best_insect_model_2048D.pkl",
    "model_100d": "models/best_insect_model_100D.pkl",
    "pca": "models/pca_transformer_100D.pkl",
}


@st.cache_resource
def load_app_assets():
    clf_2048 = joblib.load(PATHS["model_2048"])
    clf_100d = joblib.load(PATHS["model_100d"])
    pca_transformer = joblib.load(PATHS["pca"])

    dataset_embeddings = np.load(PATHS["embeddings"])
    dataset_labels = np.load(PATHS["labels"])
    umap_df = pd.read_csv(PATHS["umap"])

    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return (
        clf_2048,
        clf_100d,
        pca_transformer,
        resnet,
        preprocess,
        dataset_embeddings,
        dataset_labels,
        umap_df,
    )


(
    clf_2048,
    clf_100d,
    pca_transformer,
    resnet,
    preprocess,
    dataset_embeddings,
    dataset_labels,
    umap_df,
) = load_app_assets()


PEST_POLLINATOR_DB = {
    "Pest": [1, 5, 12, 18, 22, 31, 36],
    "Pollinator": [0, 2, 4, 8, 15, 20, 25, 30, 44, 79, 86],
}


def get_ecological_status(class_id):
    if class_id in PEST_POLLINATOR_DB["Pest"]:
        return (
            "Pest",
            "Harmful to Agricultural Crops",
            "badge-pest",
        )
    elif class_id in PEST_POLLINATOR_DB["Pollinator"]:
        return (
            "Pollinator",
            "Beneficial for Biodiversity",
            "badge-pollinator",
        )
    else:
        return (
            "Neutral / Beneficial",
            "Ecological Monitor Species",
            "badge-neutral",
        )


with st.sidebar:
    st.markdown("## 🌿 PestOrPollinator")
    st.markdown('<span class="accent-script">latent ecology engine</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Engine Options")

    selected_engine = st.radio(
        "Classification Architecture:",
        ["Full Embeddings (2048D)", "Compressed PCA (100D)"],
        help="Toggle between full 2,048D feature vector space and compressed 100D PCA vector space.",
    )

    umap_color_mode = st.radio(
        "UMAP Latent Space View:",
        ["True Biological Species", "KMeans Visual Clusters"],
        help="Switch between biological taxonomy labels and unsupervised visual cluster assignments.",
    )

    st.markdown("---")
    st.markdown("### Pipeline Mechanics")
    st.caption("• **Feature Extraction:** PyTorch ResNet50 (2048D)")
    st.caption("• **Classification:** Logistic Regression")
    st.caption("• **Retrieval:** Cosine Similarity Metric")
    st.caption("• **Topology:** Topological UMAP Reduction")


st.markdown(
    """
    <div class="hero-card">
        <span class="accent-script">field notes from the latent space</span>
        <h1 style="margin: 10px 0 0 0; font-size: 2.5rem; font-weight: 700;">Ecological Pest & Pollinator Intelligence</h1>
        <p style="margin-top: 10px; font-size: 1.05rem; color: #C4D4BE;">
            Upload an image to identify species, analyze visual similarity neighbors, and locate coordinates in the latent embedding universe.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Select or drop a butterfly/insect photo...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Extracting deep 2,048-dimensional features..."):
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            raw_embedding = resnet(img_tensor).numpy().flatten()

    if selected_engine == "Compressed PCA (100D)":
        eval_embedding = pca_transformer.transform([raw_embedding])
        prediction = clf_100d.predict(eval_embedding)[0]
        probs = clf_100d.predict_proba(eval_embedding)[0]
    else:
        eval_embedding = [raw_embedding]
        prediction = clf_2048.predict(eval_embedding)[0]
        probs = clf_2048.predict_proba(eval_embedding)[0]

    confidence = np.max(probs) * 100
    eco_role, eco_desc, badge_style = get_ecological_status(prediction)

    col_img, col_info = st.columns([1, 1.4], gap="large")

    with col_img:
        st.markdown('<div class="nature-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded Sample")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(
            f"""
            <div class="nature-card">
                <span class="{badge_style}">{eco_role}</span>
                <p style="margin-top: 14px; margin-bottom: 0px;" class="metric-label">Predicted Species Identification</p>
                <h2 class="mono" style="margin-top: 4px; font-size: 1.9rem; color: var(--text-primary); font-weight: 700;">Class #{prediction}</h2>
                <hr style="border: none; border-top: 1px solid var(--card-border); margin: 16px 0;">
                <div style="display: flex; gap: 32px;">
                    <div>
                        <p class="metric-label">Model Confidence</p>
                        <p class="metric-value">{confidence:.1f}%</p>
                    </div>
                    <div>
                        <p class="metric-label">Ecological Profile</p>
                        <p style="font-weight: 600; color: var(--text-primary); font-size: 1.05rem; margin-top: 8px;">{eco_desc}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Top 5 Visually Similar Matches")
    st.caption(
        "Retrieved using cosine similarity on raw 2,048-dimensional embeddings."
    )

    similarities = cosine_similarity([raw_embedding], dataset_embeddings)[0]
    top_5_idx = np.argsort(similarities)[-5:][::-1]

    cols = st.columns(5)
    for i, idx in enumerate(top_5_idx):
        match_class = dataset_labels[idx]
        sim_score = similarities[idx]
        match_role, _, match_badge = get_ecological_status(match_class)

        with cols[i]:
            st.markdown(
                f"""
                <div class="nature-card" style="text-align: center; padding: 16px;">
                    <p class="metric-label">Match #{i+1}</p>
                    <h4 class="mono" style="margin: 4px 0; color: var(--text-primary);">Class {match_class}</h4>
                    <p class="mono" style="font-weight: 700; color: var(--accent); font-size: 1.1rem;">{sim_score:.3f}</p>
                    <span class="{match_badge}" style="font-size: 0.7rem; padding: 4px 10px;">{match_role}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Topological Latent Space Map")
    st.caption(
        "Interactive 2D UMAP projection showing where high-dimensional visual embeddings cluster."
    )

    color_column = (
        "Class"
        if umap_color_mode == "True Biological Species"
        else "KMeans_Cluster"
    )

    fig = px.scatter(
        umap_df,
        x="UMAP_1",
        y="UMAP_2",
        color=umap_df[color_column].astype(str),
        opacity=0.75,
        hover_data=["Class", "KMeans_Cluster"],
        color_discrete_sequence=[
            "#A3FF6B", "#6BC7FF", "#FF6B5B", "#FFD56B", "#B26BFF",
            "#6BFFD1", "#FF9F6B", "#8FA8FF", "#D1FF6B", "#FF6BC1",
        ],
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0F1A0C",
        font={"family": "Space Grotesk", "color": "#EAF2E4"},
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="#1E2E1A", zerolinecolor="#1E2E1A"),
        yaxis=dict(showgrid=True, gridcolor="#1E2E1A", zerolinecolor="#1E2E1A"),
    )

    st.markdown('<div class="nature-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="nature-card" style="text-align: center; padding: 56px;">
            <span class="accent-script">the canopy is quiet</span>
            <h3 style="color: var(--text-primary); margin: 10px 0 8px 0;">No image uploaded yet</h3>
            <p style="color: var(--text-muted); margin: 0;">Upload a JPEG or PNG photo above to execute classification and visual similarity search.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )