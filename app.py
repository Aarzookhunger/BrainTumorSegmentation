import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image
import io
import time

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ----------------- GLOBAL STYLES (medical look) -----------------
st.markdown("""
<style>
body, .stApp {
    background: #e5edf5;
}

/* Main card */
.main .block-container {
    background: #f9fafb;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    border: 1px solid #d1d5db;
}

/* Navbar */
.top-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #0f172a;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #1f2937;
    padding: 0.65rem 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.nav-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.nav-logo {
    font-size: 1.4rem;
}
.nav-title {
    color: #e5e7eb;
    font-weight: 600;
    font-size: 1.05rem;
}
.nav-subtitle {
    color: #9ca3af;
    font-size: 0.72rem;
}
.nav-right {
    display: flex;
    gap: 0.7rem;
    font-size: 0.9rem;
}
.nav-link {
    color: #9ca3af;
    text-decoration: none;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    border: 1px solid transparent;
}
.nav-link:hover {
    color: #e5e7eb;
    border-color: #374151;
    background: #111827;
}

/* Typography */
h1 {
    color: #111827;
    text-align: left;
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}
.caption { 
    color: #4b5563; 
    text-align: left;
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}

/* Patient section card */
.patient-card {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #dbe2ea;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

/* Compact file uploader styling */
[data-testid="stFileUploader"] > div:first-child {
    display: none; /* hide default label */
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 999px;
    padding: 0.35rem 0.75rem;
    border: 1px dashed #93c5fd;
    background-color: #f3f4ff;
}
[data-testid="stFileUploaderDropzone"] > div {
    justify-content: center;
}
[data-testid="stFileUploaderDropzone"] span {
    font-size: 0.8rem;
}

/* Download buttons */
.stDownloadButton button {
    background-color: #2563eb;
    color: #f9fafb;
    border-radius: 8px;
    border: none;
    padding: 0.55rem 1.3rem;
    font-weight: 500;
    margin-top: 0.4rem;
}
.stDownloadButton button:hover {
    background: #1d4ed8;
}

/* Session history cards */
.history-card {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #d1d5db;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.8rem;
}
.history-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.4rem;
}
.hist-patient-name {
    font-weight: 600;
    color: #111827;
}
.hist-patient-meta {
    font-size: 0.75rem;
    color: #6b7280;
}
.hist-tumor {
    font-size: 0.85rem;
    font-weight: 600;
    color: #059669;
}

/* Misc */
hr {border-top: 1px solid #d1d5db;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- NAVBAR -----------------
st.markdown("""
<div class="top-nav">
  <div class="nav-left">
    <span class="nav-logo">🧠</span>
    <div>
      <div class="nav-title">NeuroScan — Tumor Segmentation</div>
      <div class="nav-subtitle">Clinical Decision Support · MRI Brain</div>
    </div>
  </div>
  <div class="nav-right">
    <a href="#analyzer" class="nav-link">Analyzer</a>
    <a href="#session-history" class="nav-link">Session history</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------- MODEL -----------------
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

def predict_tumor(image_path):
    pre_img = preprocess_image(image_path)
    input_img = np.expand_dims(pre_img, axis=0)
    start_time = time.time()
    pred_mask = model.predict(input_img, verbose=0)[0]
    runtime = time.time() - start_time
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    orig = cv2.imread(image_path)
    orig_resized = cv2.resize(orig, (256, 256))

    # Soft blending (no harsh patch)
    color_layer = np.zeros_like(orig_resized)
    color_layer[:] = (180, 130, 255)  # soft magenta (BGR) for medical overlay
    blended_full = cv2.addWeighted(orig_resized, 0.70, color_layer, 0.30, 0)

    overlay = orig_resized.copy()
    mask_bool = pred_mask.squeeze() == 255
    overlay[mask_bool] = blended_full[mask_bool]

    # Thin contour to delineate tumor border
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return orig_resized, pred_mask.squeeze(), overlay, runtime

# ----------------- SESSION STATE -----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------- HEADER -----------------
st.markdown('<a name="analyzer"></a>', unsafe_allow_html=True)
st.title("Brain Tumor Segmentation")
st.markdown(
    '<div class="caption">Upload MRI brain scans, add patient details, and review segmented tumor regions with slice-wise tumor area estimation.</div>',
    unsafe_allow_html=True
)

# ----------------- PATIENT DETAILS + UPLOAD -----------------
st.markdown('<div class="patient-card">', unsafe_allow_html=True)
st.write("### Patient details & MRI upload")

pd_col1, pd_col2, pd_col3 = st.columns([1.2, 1, 1.2])
with pd_col1:
    patient_name = st.text_input("Patient name", placeholder="e.g., John Doe")
    patient_id = st.text_input("Patient ID / MRN", placeholder="e.g., MRN-001")
with pd_col2:
    patient_age = st.text_input("Age", placeholder="e.g., 54")
    patient_gender = st.selectbox("Gender", ["-", "Male", "Female", "Other"], index=0)
with pd_col3:
    patient_notes = st.text_area("Clinical notes", placeholder="Optional clinical context (symptoms, findings)...", height=80)

# compact uploader in a small column so it doesn't span the whole line
upload_col, _ = st.columns([1.2, 2])
with upload_col:
    uploaded_files = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multiupload",
        label_visibility="collapsed"
    )
    st.caption("**Upload MRI** (multiple slices allowed)")

st.markdown('</div>', unsafe_allow_html=True)

current_patient = {
    "name": (patient_name or "").strip(),
    "id": (patient_id or "").strip(),
    "age": (patient_age or "").strip(),
    "gender": patient_gender if patient_gender != "-" else "",
    "notes": (patient_notes or "").strip()
}

# ----------------- PROCESS UPLOADED FILES -----------------
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            orig, mask, overlay, runtime = predict_tumor(tmp_path)

        tumor_pixels = np.sum(mask == 255)
        total_pixels = mask.size
        tumor_pct = (tumor_pixels / total_pixels) * 100

        st.markdown("<hr>", unsafe_allow_html=True)

        cols = st.columns([2.2, 2.2, 2.2, 1.4])

        cols[0].image(
            cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
            caption=f"Original MRI slice ({uploaded_file.name})",
            use_container_width=True
        )

        cols[1].image(
            mask,
            caption="Predicted mask",
            use_container_width=True,
            clamp=True
        )

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        cols[2].image(
            overlay_rgb,
            caption="Tumor highlight (blended)",
            use_container_width=True
        )

        # Tumor metric + compact patient summary for this slice
        with cols[3]:
            st.metric(
                label="Tumor area (this slice)",
                value=f"{tumor_pct:.2f}%",
                help="Approximate percentage of this MRI slice occupied by tumor."
            )
            st.write("---")
            st.markdown(
                f"**Patient:** {current_patient.get('name') or 'Not specified'}<br>"
                f"**ID:** {current_patient.get('id') or '-'}<br>"
                f"**Age:** {current_patient.get('age') or '-'}",
                unsafe_allow_html=True
            )

        st.info(f"Processing time: {runtime:.2f}s", icon="ℹ️")

        # Downloads
        mask_img = Image.fromarray(mask)
        overlay_img = Image.fromarray(overlay_rgb)
        buf_mask = io.BytesIO()
        buf_overlay = io.BytesIO()
        mask_img.save(buf_mask, format="PNG")
        overlay_img.save(buf_overlay, format="PNG")
        c1, c2 = st.columns(2)
        c1.download_button("Download mask", buf_mask.getvalue(),
                           f"mask_{uploaded_file.name}", "image/png")
        c2.download_button("Download overlay", buf_overlay.getvalue(),
                           f"overlay_{uploaded_file.name}", "image/png")

        # Save to session history with patient details
        st.session_state.history.append({
            "name": uploaded_file.name,
            "original": cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
            "mask": mask.copy(),
            "overlay": overlay_rgb,
            "tumor_pct": f"{tumor_pct:.2f}%",
            "tumor_pct_value": tumor_pct,
            "patient": current_patient.copy()
        })

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------- SESSION HISTORY (with patient details first) -----------------
st.markdown('<a name="session-history"></a>', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("## Session history")

    for i, item in enumerate(st.session_state.history[::-1]):
        patient = item.get("patient", {}) or {}
        p_name = patient.get("name") or "Unknown patient"
        p_id = patient.get("id") or "N/A"
        p_age = patient.get("age") or "N/A"
        p_gender = patient.get("gender") or "-"
        p_notes = patient.get("notes") or ""

        st.markdown('<div class="history-card">', unsafe_allow_html=True)

        # Patient + tumor summary header
        st.markdown(
            f"""
            <div class="history-header">
              <div>
                <div class="hist-patient-name">{p_name}</div>
                <div class="hist-patient-meta">
                    ID: {p_id} &nbsp;·&nbsp; Age: {p_age} &nbsp;·&nbsp; Gender: {p_gender}
                </div>
              </div>
              <div class="hist-tumor">
                Tumor area (slice): {item['tumor_pct']}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if p_notes:
            st.markdown(
                f"<div class='hist-patient-meta'>Notes: {p_notes}</div>",
                unsafe_allow_html=True
            )

        # MRI thumbnails
        ch = st.columns([2, 2, 2])
        ch[0].image(item["original"],
                    caption=f"Original ({item['name']})",
                    use_container_width=True)
        ch[1].image(item["mask"],
                    caption="Mask",
                    use_container_width=True,
                    clamp=True)
        ch[2].image(item["overlay"],
                    caption="Overlay",
                    use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("No history yet. Process images to view session history.")
