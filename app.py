# app.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tempfile
import time
from datetime import datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------
# Styles (Healthcare theme + skeleton loader)
# ---------------------------
st.markdown("""
<style>
:root{
  --primary: #0b6ef6;
  --primary-dark:#085cc2;
  --accent:#2dd4bf;
  --bg:#eef6fb;
  --card:#ffffff;
  --muted:#6b7785;
  --danger:#ef4444;
  --ok:#10b981;
  --radius:12px;
  --shadow: 0 6px 30px rgba(11,30,60,0.08);
}

/* App background and fonts */
body, .stApp { background: var(--bg); font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }

/* NAVBAR */
.navbar{
  position: sticky; top:0; z-index:9999;
  background: linear-gradient(90deg, rgba(11,30,60,1) 0%, rgba(6,21,56,1) 100%);
  padding: 14px 26px; display:flex; align-items:center; gap:18px; border-radius:0 0 14px 14px; box-shadow: var(--shadow);
}
.nav-title{ color: white; font-weight:700; font-size:1.45rem; letter-spacing:0.4px; }
.nav-link{ margin-left: 18px; color:#bcd4ff; text-decoration:none; padding:6px 12px; border-radius:8px; font-weight:600; }
.nav-link:hover{ background: rgba(255,255,255,0.06); color: #fff; }

/* Page headings and cards */
.page-title{ color:#0b2336; font-size:1.35rem; font-weight:700; margin-top:0.6rem; }
.card{ background:var(--card); padding:14px; border-radius:var(--radius); box-shadow: var(--shadow); margin-bottom:12px; }
.caption{ color:var(--muted); margin-bottom:10px; }

/* Patient banner for doctor (prominent) */
.patient-banner{
  display:flex; gap:12px; align-items:center; justify-content:space-between;
  padding:12px; border-radius:10px; background: linear-gradient(90deg, rgba(245,247,250,1) 0%, rgba(234,241,249,1) 100%);
  border: 1px solid #e6eef9; margin-bottom: 12px;
}
.patient-left{ display:flex; gap:12px; align-items:center;}
.patient-name{ font-weight:700; font-size:1.12rem; color:#07203a; }
.patient-meta{ color: #475569; font-size:0.92rem; }
.detect-badge {
  padding:10px 14px; border-radius:10px; font-weight:700; font-size:0.98rem;
}
.badge-danger{ background: linear-gradient(90deg,var(--danger), #c53030); color: white; }
.badge-ok{ background: linear-gradient(90deg,var(--ok), #059669); color: white; }

/* Metric box */
.metric-box{ background:#0b6ef6; color:#fff; padding:12px; border-radius:10px; text-align:center; font-weight:700; }

/* Download buttons */
.stDownloadButton button{ background:var(--primary); color:#fff; border-radius:8px; padding:8px 12px; border:none; font-weight:700; }
.stDownloadButton button:hover{ background:var(--primary-dark); }

/* Gallery thumbnails */
.thumb{ border-radius:8px; box-shadow: 0 6px 18px rgba(6,21,56,0.06); }

/* Skeleton loader */
.skel-row{ display:flex; gap:14px; align-items:center; }
.skel{ width:100%; height:150px; border-radius:10px; background: linear-gradient(90deg,#f3f6f9,#e6edf9,#f3f6f9); background-size:200% 100%; animation: shd 1.2s linear infinite; }
.skel.small{ height:60px; }
@keyframes shd { 0%{background-position:200% 0;} 100%{background-position:-200% 0;} }

/* small text hint */
.hint{ color:#5b6b78; font-size:0.92rem; }

/* input styling */
input, textarea, .stTextInput>div>div>input { border-radius:8px; padding:10px; border:1px solid #d6e6fb; background:white; color:#0b2336 !important; }

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Navbar
# ---------------------------
st.markdown("""
<div class="navbar">
  <div style="display:flex;align-items:center;gap:12px;">
    <div class="nav-title">Brain Tumor Segmentation Portal</div>
    <a class="nav-link" href="#upload">Upload</a>
    <a class="nav-link" href="#gallery">Gallery</a>
    <a class="nav-link" href="#about">About</a>
  </div>
  <div style="margin-left:auto; display:flex; gap:8px; align-items:center;">
    <div style="color:#cfe8ff; font-weight:600; font-size:0.95rem;">Clinic AI</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Model Loading (cached)
# ---------------------------
@st.cache_resource
def load_segmentation_model(path="final_model.keras"):
    # Note: keep compile=False to avoid optimizer issues if model saved without compiling
    return load_model(path, compile=False)

# Try to load the model but don't crash UI if missing: show warning
model = None
try:
    model = load_segmentation_model()
except Exception as e:
    st.warning("Model could not be loaded automatically. Make sure 'final_model.keras' is present. App will still run but predictions will fail until model is available.")
    # print exception in sidebar for debug
    st.write(f"<details><summary>Model load error (click)</summary><pre>{e}</pre></details>", unsafe_allow_html=True)

# ---------------------------
# Image processing & prediction functions
# ---------------------------
def preprocess_image(image_path, size=(256,256)):
    """Load, CLAHE on L channel, blur and resize, return float32 scaled [0,1]."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Couldn't read image")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    lab = cv2.merge((clahe, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

def blend_overlay(image, mask, alpha=0.37):
    """Create a soft blue overlay where mask==255 on the RGB image (image assumed BGR)."""
    color = np.array([15, 159, 253], dtype=np.uint8)  # BGR color
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 255] = color
    blended = cv2.addWeighted(image, 1.0, mask_rgb, alpha, 0)
    return blended

def predict_with_blend(image_path):
    """Return orig_resized (BGR), mask (uint8 0/255), overlay (BGR), runtime (s).
       If model is None, returns zero mask and large runtime placeholder.
    """
    pre_img = preprocess_image(image_path)
    input_img = np.expand_dims(pre_img, axis=0)
    start_time = time.time()
    if model is None:
        # fallback: generate an empty mask
        pred_mask = np.zeros((256,256,1), dtype=np.uint8)
        runtime = 0.0
    else:
        pred_mask = model.predict(input_img, verbose=0)[0]
        runtime = time.time() - start_time
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    orig = cv2.imread(image_path)
    orig_resized = cv2.resize(orig, (256,256))
    overlay = blend_overlay(orig_resized, pred_mask.squeeze(), alpha=0.37)
    return orig_resized, pred_mask.squeeze(), overlay, runtime

# ---------------------------
# Session state init
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Top page content: patient input
# ---------------------------
st.markdown('<div id="upload"></div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Patient & Scan Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="caption">Upload one or more MRI images (jpg/png). Use the anonymize toggle in the sidebar if required.</div>', unsafe_allow_html=True)

# sidebar controls
with st.sidebar:
    st.header("Options")
    anonymize = st.checkbox("Anonymize gallery (hide patient names)", value=False)
    st.write("Detection threshold (percent of image area). Increase for stricter detection.")
    # default threshold 0.2% of image area
    detect_thresh_pct = st.number_input("Tumor detection threshold (%)", min_value=0.0, max_value=100.0, value=0.2, step=0.05, format="%.2f")
    st.write("---")
    st.write("App info")
    st.caption("Created for demo. Always get clinical confirmation.")

cols_patient = st.columns([2,2,1.3,2.7])
with cols_patient[0]:
    pname = st.text_input("Patient Name", key="pname")
with cols_patient[1]:
    pid = st.text_input("Patient ID", key="pid")
with cols_patient[2]:
    age = st.text_input("Age", key="age")
with cols_patient[3]:
    note = st.text_input("Notes (scan date, clinical info)", key="note")

uploaded_files = st.file_uploader("Upload MRI image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True)

# ---------------------------
# Helper: show skeleton while processing
# ---------------------------
def show_skeleton(target, message="Analyzing image..."):
    """Renders an animated skeleton loader into the provided placeholder (st.empty())."""
    target.markdown(f"""
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
         <div style="font-weight:700; font-size:1.05rem;">{message}</div>
         <div class="hint">This may take a few seconds</div>
      </div>
      <div class="skel-row">
         <div class="skel" style="flex:2;"></div>
         <div style="flex:1; display:flex; flex-direction:column; gap:10px;">
            <div class="skel small"></div>
            <div class="skel small"></div>
            <div class="skel small"></div>
         </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Processing uploaded files
# ---------------------------
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        # write uploaded file to temp file (cv2 needs a path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # patient metadata
        patient_data = {
            "name": pname.strip() if pname else "N/A",
            "id": pid.strip() if pid else "N/A",
            "age": age.strip() if age else "N/A",
            "note": note.strip() if note else "",
            "filename": uploaded_file.name,
            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # placeholder area for skeleton -> then results
        placeholder = st.empty()
        show_skeleton(placeholder, message=f"Analyzing {uploaded_file.name} ...")

        # perform prediction
        try:
            orig, mask, overlay, runtime = predict_with_blend(tmp_path)
        except Exception as e:
            placeholder.markdown(f"<div class='card'><b>Error processing {uploaded_file.name}:</b><pre>{e}</pre></div>", unsafe_allow_html=True)
            continue

        # compute tumor area metrics
        tumor_px = int(np.sum(mask == 255))
        pct = (tumor_px / mask.size) * 100
        px_str = f"{tumor_px:,}"
        pct_str = f"{pct:.3f}%"

        # decide detection (use threshold from sidebar)
        detected = pct >= detect_thresh_pct
        # doctor-friendly banner + results
        banner_html = f"""
        <div class="patient-banner card">
           <div class="patient-left">
              <div>
                <div class="patient-name">{patient_data['name']}</div>
                <div class="patient-meta">ID: {patient_data['id']} • Age: {patient_data['age']} • File: {patient_data['filename']}</div>
                <div class="patient-meta">Notes: {patient_data['note']}</div>
              </div>
           </div>
           <div style="display:flex; gap:12px; align-items:center;">
              <div style="text-align:right;">
                 <div style="color:#334155; font-size:0.9rem;">Tumor Area</div>
                 <div style="font-weight:800; font-size:1.35rem;">{pct_str}</div>
              </div>
              <div>
                 <div class="detect-badge {'badge-danger' if detected else 'badge-ok'}">
                    {'Tumor detected' if detected else 'No tumor detected'}
                 </div>
              </div>
           </div>
        </div>
        """
        # replace skeleton with actual results
        with placeholder.container():
            st.markdown(banner_html, unsafe_allow_html=True)

            # result images row
            row = st.columns([2,2,2,1.6])
            row[0].image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original Scan", use_container_width=True)
            # mask display: ensure single-channel grayscale visualization
            row[1].image(mask, caption="Tumor Mask", use_container_width=True, clamp=True)
            row[2].image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="AI Overlay", use_container_width=True)

            # side metrics and download
            hover_str = (
                f"Name: {patient_data['name']} | ID: {patient_data['id']} | Age: {patient_data['age']} | "
                f"Date: {patient_data['when']}"
            )
            row[3].markdown(
                f'<div title="{hover_str}" class="metric-box">{pct_str}<div style="font-size:0.78rem; font-weight:600; margin-top:6px;">Tumor Area</div></div>',
                unsafe_allow_html=True
            )

            st.info(f"Processing time: {runtime:.2f}s", icon="⚡")

            # prepare downloads
            mask_img = Image.fromarray(mask)
            overlay_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            buf_mask = io.BytesIO(); buf_overlay = io.BytesIO()
            mask_img.save(buf_mask, format="PNG"); overlay_img.save(buf_overlay, format="PNG")
            dwn_cols = st.columns([1,1.2])
            dwn_cols[0].download_button("Download Mask", buf_mask.getvalue(), f"mask_{uploaded_file.name}", "image/png")
            dwn_cols[1].download_button("Download Overlay", buf_overlay.getvalue(), f"overlay_{uploaded_file.name}", "image/png")

        # push record into session history
        st.session_state.history.append({
            "patient": patient_data,
            "original": cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
            "mask": mask.copy(),
            "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            "tumor_pct": pct_str,
            "tumor_px": tumor_px,
            "detected": detected,
            "summary": (
                f"File: {uploaded_file.name}\nPatient: {patient_data['name']} (ID: {patient_data['id']}, Age: {patient_data['age']})\n"
                f"Tumor Area: {pct_str} ({px_str} pixels)\nDate: {patient_data['when']}\nNotes: {patient_data['note']}\n"
            )
        })

# ---------------------------
# Session Gallery
# ---------------------------
st.markdown('<div id="gallery"></div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Session Gallery</div>', unsafe_allow_html=True)
if st.session_state.history:
    # display latest first
    for i, entry in enumerate(reversed(st.session_state.history)):
        h = st.columns([2.2,2.2,2.2,1.6])
        pat = entry["patient"]
        # anonymize option: hide name if requested in sidebar
        display_name = "ANON" if anonymize else pat['name']
        caption = f"{display_name} ({pat['id']})"
        h[0].image(entry["original"], caption=caption, width=120)
        h[1].image(entry["mask"], caption="Mask", width=120, clamp=True)
        h[2].image(entry["overlay"], caption="Overlay", width=120)
        tt = (f"{pat['name']} | ID: {pat['id']} | Age: {pat['age']} | Notes: {pat['note']} | Date: {pat['when']}")
        badge_style = "badge-danger" if entry.get("detected", False) else "badge-ok"
        h[3].markdown(f"""
            <div title="{tt}"
            style="background:#ffffff;padding:0.8rem;border-radius:0.6em;
            color:#0b2336;text-align:center;font-size:1.03rem;font-weight:700;margin-top:0.2rem;">
            <div style="font-size:1.02rem;">{entry['tumor_pct']}</div>
            <div style="font-size:0.78rem; color:#475569; margin-top:4px;">Tumor Area</div>
            <div style="margin-top:8px;">
              <span class="detect-badge {badge_style}" style="padding:6px 10px; font-size:0.85rem;">
                {"Detected" if entry.get("detected", False) else "Not detected"}
              </span>
            </div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown('<div class="card"><div class="hint">No session results yet — upload images above to run a scan.</div></div>', unsafe_allow_html=True)

# ---------------------------
# About section
# ---------------------------
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <div style="font-weight:800; font-size:1.1rem;">About this tool</div>
      <div style="color:#475569; margin-top:6px;">This is a demonstration-grade application for brain tumor segmentation. It uses a deep learning model to highlight suspected tumor regions and provide area estimates. Results are for clinical-assistive use and must be reviewed and confirmed by a qualified radiologist.</div>
    </div>
    <div style="text-align:right;">
      <div style="font-weight:700; color:#0b6ef6;">Clinic AI</div>
      <div style="color:#6b7280; font-size:0.9rem;">v1.0 • Demo</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
