import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tempfile
import time
from datetime import datetime

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")

st.markdown("""
<style>
body, .stApp {background: #141927;}
.appbar {
    background: #181d2f;
    padding: 1.1rem 0.4rem 0.9rem 1.1rem;
    border-radius: 0px 0px 12px 12px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.14);
    margin-bottom: 1.7rem;
    display: flex; align-items:center; gap:1.5rem;
}
.appbar-title {
    color: #50d7f7;
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.nav-link {
    color: #b0bddb;
    margin-left: auto;
    margin-right: 1.5rem;
    font-size: 0.97rem;
    padding: 0.25rem 1.2rem;
    background: transparent;
    text-decoration: none;
    transition: background 0.19s;
    border-radius: 6px;
}
.nav-link:hover { background: #26335b; color: #fff;}
.cardrow {margin-top:0; margin-bottom: 1.6rem;}
.stDownloadButton button {
    background-color: #3b82f6;
    color: #eee;
    border-radius: 7px;
    border: none;
    padding: 0.55rem 1.3rem;
    font-weight: 500;
    margin-top: 0.4rem;}
.stDownloadButton button:hover {
    background: linear-gradient(90deg,#486eb7,#3493da);
    color: #fff;}
hr {border-top: 2px solid #232544;}
input, textarea, .stTextInput>div>div>input {
    background: #191c27!important;
    border: 1.5px solid #314463;
    color: #e8edf5!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="appbar">
  <div class="appbar-title">Brain Tumor Segmentation Portal</div>
  <a href="#gallery" class="nav-link">Session Gallery</a>
  <a href="#about" class="nav-link">About</a>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_segmentation_model():
    return load_model("final_model.keras", compile=False)

model = load_segmentation_model()

def preprocess_image(image_path, size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE().apply(l)
    lab = cv2.merge((clahe, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

def blend_overlay(image, mask, alpha=0.37): # soft blend mask onto image
    color = np.array([15, 159, 253], dtype=np.uint8)
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 255] = color
    blended = cv2.addWeighted(image, 1, mask_rgb, alpha, 0)
    return blended

def predict_with_blend(image_path):
    pre_img = preprocess_image(image_path)
    input_img = np.expand_dims(pre_img, axis=0)
    start_time = time.time()
    pred_mask = model.predict(input_img, verbose=0)[0]
    runtime = time.time() - start_time
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    orig = cv2.imread(image_path)
    orig_resized = cv2.resize(orig, (256, 256))
    overlay = blend_overlay(orig_resized, pred_mask.squeeze(), alpha=0.37)
    return orig_resized, pred_mask.squeeze(), overlay, runtime

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    '<div class="caption">Automatic and private brain tumor detection for MRI scans. '
    'Enter patient details below, upload one or more scans, and receive enhanced, '
    'color-coded results with one click.</div>',
    unsafe_allow_html=True
)

cols_patient = st.columns([2, 2, 2, 3])
with cols_patient[0]:
    pname = st.text_input("Patient Name", key="pname")
with cols_patient[1]:
    pid = st.text_input("Patient ID", key="pid")
with cols_patient[2]:
    age = st.text_input("Age", key="age")
with cols_patient[3]:
    note = st.text_input("Notes (e.g. scan date, clinical info)", key="note")

uploaded_files = st.file_uploader(
    "Upload MRI", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi"
)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        patient_data = {
            "name": pname.strip() if pname else "N/A",
            "id": pid.strip() if pid else "N/A",
            "age": age.strip() if age else "N/A",
            "note": note.strip() if note else "",
            "filename": uploaded_file.name,
            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            orig, mask, overlay, runtime = predict_with_blend(tmp_path)

        tumor_px = int(np.sum(mask == 255))
        pct = (tumor_px / mask.size) * 100
        px_str = f"{tumor_px:,}"
        pct_str = f"{pct:.2f}%"

        st.markdown(f'<hr class="cardrow">', unsafe_allow_html=True)
        card = st.columns([2.5,2.5,2.5,2.2])
        card[0].image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        card[1].image(mask, caption="Tumor Mask", use_container_width=True, clamp=True)
        card[2].image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Enhanced Overlay", use_container_width=True)
        hover_str = (
            f"Name: {patient_data['name']} | ID: {patient_data['id']} | Age: {patient_data['age']} | "
            f"Date: {patient_data['when']}"
        )
        # Tumor metric with patient info in tooltip
        card[3].markdown(
            f'<div title="{hover_str}" style="background:#222f44;padding:1.1rem;border-radius:0.7em;'
            f'color:#74fee1;text-align:center;font-size:1.23rem;font-weight:600">'
            f"Tumor<br>Area<br><span>{pct_str}</span></div>",
            unsafe_allow_html=True
        )
        st.info(f"Processing time: {runtime:.2f}s")

        mask_img = Image.fromarray(mask)
        overlay_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        buf_mask = io.BytesIO()
        buf_overlay = io.BytesIO()
        mask_img.save(buf_mask, format="PNG")
        overlay_img.save(buf_overlay, format="PNG")
        dwn_cols = st.columns([1,1.2])
        dwn_cols[0].download_button("Download Mask", buf_mask.getvalue(), f"mask_{uploaded_file.name}", "image/png")
        dwn_cols[1].download_button("Download Overlay", buf_overlay.getvalue(), f"overlay_{uploaded_file.name}", "image/png")

        summary = (
            f"File: {uploaded_file.name}\nPatient: {patient_data['name']} (ID: {patient_data['id']}, Age: {patient_data['age']})\n"
            f"Tumor Area: {pct_str} ({px_str} pixels)\nDate: {patient_data['when']}\nNotes: {patient_data['note']}\n"
        )

        st.session_state.history.append({
            "patient": patient_data,
            "original": cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
            "mask": mask.copy(),
            "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            "tumor_pct": pct_str,
            "summary": summary
        })

st.markdown('<hr id="gallery">', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown('<h3 style="color:#97e3f7;margin-top:2.3rem;">Session Gallery</h3>', unsafe_allow_html=True)
    for i, entry in enumerate(reversed(st.session_state.history)):
        h = st.columns([2.1,2.1,2.2,2])
        pat = entry["patient"]
        h[0].image(entry["original"], caption=f"{pat['name']} ({pat['id']})", width=110)
        h[1].image(entry["mask"], caption="Mask", width=110, clamp=True)
        h[2].image(entry["overlay"], caption="Overlay", width=110)
        tt = (f"{pat['name']} | ID: {pat['id']} | Age: {pat['age']} | "
              f"Notes: {pat['note']} | Date: {pat['when']}")
        h[3].markdown(f"""
            <div title="{tt}"
            style="background:#222f44;padding:0.95rem;border-radius:0.6em;
            color:#7dffdb;text-align:center;font-size:1.08rem;font-weight:600;margin-top:0.2rem;">
            <span style="font-size:1.03em;">{entry['tumor_pct']}</span><br>
            <span style="font-size:0.75em;">Tumor Area</span></div>
        """, unsafe_allow_html=True)
        # Tooltip shows patient details if hovered

st.markdown('<hr id="about">', unsafe_allow_html=True)
st.markdown(
    '<div style="color:#b6bdc6;font-size:0.96em;padding-bottom:10px;">'
    'Created for demonstration and educational purposes. This tool assists MRI review by highlighting suspected tumor regions. Automated results should always be interpreted by a clinical expert.</div>',
    unsafe_allow_html=True
)
