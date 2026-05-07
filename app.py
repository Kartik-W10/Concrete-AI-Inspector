import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid AI Structural Inspector",
    page_icon="",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️  Machine Vision ProjectS")
st.subheader("Deep Learning (Semantic) vs. Traditional (Gradient-Based) Analysis")
st.markdown("---")

# --- Model Loading ---
MODEL_PATH = "weights/best.pt"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

model = load_model()

# --- Sidebar: Vision Pipeline Configuration ---
st.sidebar.header("🛠️ Vision Pipeline")

# 1. AI Settings
st.sidebar.subheader("1. AI Model (Semantic)")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25)

# 2. Traditional MV Settings (Canny)
st.sidebar.markdown("---")
st.sidebar.subheader("2. Traditional MV (Canny)")
enable_canny = st.sidebar.toggle("Enable Canny Comparison", value=True)
canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)

# 3. Preprocessing (DIP)
st.sidebar.markdown("---")
st.sidebar.subheader("3. Preprocessing")
apply_bilateral = st.sidebar.toggle("Bilateral Filtering (Edge-Preserve)", value=False)
if apply_bilateral:
    d = st.sidebar.slider("Diameter (d)", 1, 15, 9)
    s_color = st.sidebar.slider("Sigma Color", 10, 150, 75)
    s_space = st.sidebar.slider("Sigma Space", 10, 150, 75)

# --- Main App Logic ---
if model is None:
    st.error(f"Weights not found! Ensure 'best.pt' is in the 'weights' folder.")
else:
    uploaded_file = st.file_uploader("Upload Surface Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        # Load and Prepare Images
        raw_img = Image.open(uploaded_file)
        img_np = np.array(raw_img.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # A. Apply Preprocessing
        if apply_bilateral:
            img_bgr = cv2.bilateralFilter(img_bgr, d, s_color, s_space)

        # B. AI Inference (Semantic Segmentation)
        results = model.predict(source=img_bgr, conf=conf_threshold)
        ai_output = results[0].plot()

        # C. Traditional Inference (Canny Edge Detection)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, canny_low, canny_high)
        # Convert Canny mask to BGR for display
        canny_display = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

        # --- Display Comparison Layout ---
        col1, col2 = st.columns(2)

        with col1:
            st.info(" High-Level AI Segmentation")
            st.image(ai_output, channels="BGR", use_container_width=True, caption="YOLOv11: Learns 'What' a crack is.")

        with col2:
            if enable_canny:
                st.warning(" Low-Level Canny Detection")
                st.image(canny_display, use_container_width=True, caption="Canny: Only calculates brightness gradients.")
            else:
                st.info(" Original Input")
                st.image(img_np, use_container_width=True, caption="Source Image")

        # --- Engineering Analytics Report ---
        st.markdown("---")
        st.subheader(" Comparative Damage Analytics")

        m1, m2, m3 = st.columns(3)

        # AI Metrics
        if results[0].masks is not None:
            combined_mask = torch.any(results[0].masks.data, dim=0)
            ai_defect_pixels = torch.sum(combined_mask).item()
            ai_ratio = (ai_defect_pixels / combined_mask.numel()) * 100
            
            m1.metric("AI Defect Count", len(results[0].boxes))
            m2.metric("AI Surface Area", f"{ai_ratio:.3f}%")
            
            if ai_ratio > 5:
                m3.error("Status: CRITICAL")
            elif ai_ratio > 1:
                m3.warning("Status: MONITORING")
            else:
                m3.success("Status: NOMINAL")
        else:
            m1.metric("AI Defect Count", "0")
            m2.metric("AI Surface Area", "0.000%")
            m3.success("Status: CLEAN")

        # Traditional Metric Comparison
        canny_pixels = np.sum(canny_edges > 0)
        canny_density = (canny_pixels / canny_edges.size) * 100
        
        st.write(f"**Note:** Canny Edge Density is **{canny_density:.2f}%**. Notice how Canny detects surface pores and shadows as 'edges', whereas the AI ignores this noise.")

        # Mathematical Methodology
        with st.expander("🎓 Technical Methodology"):
            st.write("### Comparison of Methodologies")
            st.write("**1. Semantic Segmentation (YOLOv11):** Uses a deep CNN backbone to extract high-level semantic features. It understands the *context* of a crack.")
            st.write("**2. Canny Edge Detection:** A multi-stage algorithm involving Gaussian blurring, gradient calculation (Sobel), non-maximum suppression, and hysteresis thresholding.")
            st.latex(r"Damage \% = \left( \frac{\sum P_{defect}}{\sum P_{total}} \right) \times 100")

    else:
        st.info("Waiting for image upload to begin Hybrid Vision analysis...")