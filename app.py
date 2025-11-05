import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tempfile
import time
from datetime import datetime

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    layout="wide"
)

# ------------------------------------------------------------------------------
# CUSTOM CSS (Healthcare UI)
# ------------------------------------------------------------------------------
st.markdown("""
<style>

body, .stApp {
    background-color: #F3F6FA !important;
}

.navbar {
    background: #ffffff;
    padding: 1rem 2rem;
    border-bottom: 1px solid #e5e8ef;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.nav-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1C3B63;
}

.nav-links a {
    margin-left: 1.5rem;
    color: #335985;
    text-decoration: none;
    font-weight: 500;
}

.patient-card {
    background: white;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    color: #1f355b;
    font-size: 0.95rem;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1C3B63;
    margin-top: 1.8rem;
}

.result-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
}

.metric-box {
    background: #e8f1ff;
    border-left: 5px solid #4C84F7;
    padding: 1rem;
    border-radius: 10px;
    color: #1C3B63;
    font-weight: 600;
    text-align: center;
}

.upload-btn input {
    background-color: #4C84F7 !important;
    color: white !important;
    padding: 0.7rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.skeleton {
    width: 100%;
    height: 220px;
    border-radius: 12px;
    background: linear-gradient(90deg, #e2e5e7 25%, #f7f8f8 37%, #e2e5e7 63%);
    background-size: 400% 100%;
    animation: shimmer 1.4s ease infinite;
    margin-bottom: 1rem;
}

@keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# NAVBAR
# ------------------------------------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="nav-title">🧠 Brain Tumor Segmentation Portal</div>

    <div class="nav-links">
        <a href="#upload">Upload Scan</a>
        <a href="#history">History</a>
        <a href="#about">About</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODEL LOAD
# ------------------------------------------------------------------------------
@st.cache_resource
def load_segmentation_model():
    return load_model("final_model.keras", compile=False)

model = load_segmentation_model()


# ------------------------------------------------------------------------------
# IMAGE PROCESSING
# ------------------------------------------------------------------------------
def preprocess_image(image_path, size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(l)
    clahe = cv2.createCLAHE().apply(l)
    lab = cv2.merge((clahe, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def blend_overlay(image, mask, alpha=0.35):
    color = np.array([0, 128, 255], dtype=np.uint8)  # Medical blue
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 255] = color
    blended = cv2.addWeighted(image, 1, mask_rgb, alpha, 0)
    return blended


def predict_with_blend(image_path):
    pre_img = preprocess_image(image_path)
    input_img = np.expand_dims(pre_img, axis=0)
    start = time.time()
    pred_mask = model.predict(input_img, verbose=0)[0]
    runtime = time.time() - start

    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    orig = cv2.imread(image_path)
    orig_resized = cv2.resize(orig, (256, 256))

    overlay = blend_overlay(orig_resized, pred_mask.squeeze())
    return orig_resized, pred_mask.squeeze(), overlay, runtime


# ------------------------------------------------------------------------------
# PATIENT FORM SECTION
# ------------------------------------------------------------------------------
st.markdown('<div id="upload"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
pname = col1.text_input("Name")
pid = col2.text_input("Patient ID")
age = col3.text_input("Age")
note = col4.text_input("Notes")


# ------------------------------------------------------------------------------
# UPLOAD BUTTON (compact)
# ------------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------------------
# PROCESS EACH IMAGE
# ------------------------------------------------------------------------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        patient_data = {
            "name": pname or "Unknown",
            "id": pid or "N/A",
            "age": age or "N/A",
            "note": note or "",
            "filename": uploaded_file.name,
            "date": datetime.now().strftime("%d %b %Y - %H:%M")
        }

        # -------------------------------------------
        # SKELETON LOADING PLACEHOLDER
        # -------------------------------------------
        with st.spinner("Analyzing MRI Scan..."):
            sk_cols = st.columns(3)
            for c in sk_cols:
                c.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
            time.sleep(1.2)

        # -------------------------------------------
        # REAL PREDICTION
        # -------------------------------------------
        orig, mask, overlay, runtime = predict_with_blend(tmp_path)

        tumor_px = int(np.sum(mask == 255))
        pct = (tumor_px / mask.size) * 100
        pct_str = f"{pct:.2f}%"

        # -------------------------------------------
        # RESULT DISPLAY
        # -------------------------------------------
        st.markdown('<div class="section-title">Segmentation Result</div>', unsafe_allow_html=True)

        # Patient info card
        st.markdown(f"""
        <div class="patient-card">
            <b>Patient:</b> {patient_data['name']}<br>
            <b>ID:</b> {patient_data['id']}<br>
            <b>Age:</b> {patient_data['age']}<br>
            <b>Date:</b> {patient_data['date']}<br>
            <b>Notes:</b> {patient_data['note']}
        </div>
        """, unsafe_allow_html=True)

        result_cols = st.columns(3)
        result_cols[0].image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
        result_cols[1].image(mask, caption="Tumor Mask", use_container_width=True, clamp=True)
        result_cols[2].image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay", use_container_width=True)

        st.markdown(f"""
        <div class="metric-box">
            Tumor Area: {pct_str}
        </div>
        """, unsafe_allow_html=True)

        # SAVE TO HISTORY
        st.session_state.history.append({
            "patient": patient_data,
            "orig": cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
            "mask": mask.copy(),
            "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            "pct": pct_str
        })

# ------------------------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------------------------
st.markdown('<div id="history"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Session History</div>', unsafe_allow_html=True)

if st.session_state.history:
    for entry in reversed(st.session_state.history):
        colA, colB, colC, colD = st.columns([1.5, 1.5, 1.5, 2])

        colA.image(entry["orig"], caption="Original", width=150)
        colB.image(entry["mask"], caption="Mask", width=150)
        colC.image(entry["overlay"], caption="Overlay", width=150)

        pat = entry["patient"]
        colD.markdown(f"""
        <div class="patient-card" style="font-size:0.85rem;">
            <b>{pat['name']}</b><br>
            ID: {pat['id']}<br>
            Age: {pat['age']}<br>
            Tumor Area: {entry['pct']}<br>
            Notes: {pat['note']}<br>
            Date: {pat['date']}
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------------------------
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-title">About</div>
This tool provides AI-powered assistance for brain MRI tumor segmentation.
Results are for educational and diagnostic support only. Always follow clinical judgment.
""")
