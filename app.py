import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image
import io
import time

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ----------------- GLOBAL STYLES -----------------
st.markdown("""
<style>
/* Background + global text color */
body, .stApp {
    background: #e2f0ff;  /* medical bluish */
    color: #111111 !important;
}

/* Main container card */
.main .block-container {
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    border: 1px solid #e5e7eb;
}

/* Top navbar */
.top-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #ffffff;
    border-bottom: 2px solid #2563eb;
    box-shadow: 0 2px 8px rgba(15,23,42,0.12);
    padding: 0.55rem 1.6rem;
    display: flex;
    align-items: center;
}
.nav-logo {
    font-size: 1.6rem;
    margin-right: 0.6rem;
}
.nav-title {
    color: #111111;
    font-weight: 700;
    font-size: 1.1rem;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: #111111 !important;
}
h1 {
    text-align: left;
    font-size: 1.9rem !important;
    margin-bottom: 0.4rem;
}
.caption { 
    color: #111111; 
    text-align: left;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}

/* Inputs like clinical form */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    border-radius: 8px !important;
    border: 1px solid #9ca3af !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    color: #111111 !important;
    font-size: 0.85rem !important;
}

/* Patient section card */
.patient-card {
    background: #f8fbff;
    border-radius: 12px;
    border: 1px solid #d0def2;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}

/* Upload MRI – small, centered */
.upload-wrapper {
    max-width: 260px;
    margin: 0.5rem auto 0 auto;
}
[data-testid="stFileUploader"] > div:first-child {
    display: none; /* hide default label row */
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 999px !important;
    padding: 0.25rem 0.6rem !important;
    border: 1px dashed #60a5fa !important;
    background-color: #f0f6ff !important;
}
[data-testid="stFileUploaderDropzone"] > div {
    justify-content: center !important;
}
[data-testid="stFileUploaderDropzone"] span {
    display: none !important;  /* hide "Drag and drop..." + size text */
}

/* Analyze button */
.analyze-btn button {
    border-radius: 999px;
    padding: 0.3rem 1.4rem;
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
    background: #f8fbff;
    border-radius: 12px;
    border: 1px solid #d1d5db;
    padding: 0.8rem 1rem;
    margin-bottom: 0.9rem;
}
.history-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.4rem;
}
.hist-patient-name {
    font-weight: 700;
    font-size: 1.0rem;   /* bigger font */
    color: #111111;
}
.hist-patient-meta {
    font-size: 0.9rem;   /* bigger font */
    color: #111111;
}
.hist-tumor {
    font-size: 0.95rem;
    font-weight: 700;
    color: #065f46;
}

/* Misc */
hr {border-top: 1px solid #e5e7eb;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- NAVBAR -----------------
st.markdown("""
<div class="top-nav">
  <span class="nav-logo">🧠</span>
  <span class="nav-title">Brain Tumor Detection</span>
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
    """Return original, mask, and a softly blended heatmap overlay."""
    pre_img = preprocess_image(image_path)
    input_img = np.expand_dims(pre_img, axis=0)
    start_time = time.time()
    pred_mask = model.predict(input_img, verbose=0)[0]
    runtime = time.time() - start_time

    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # (256,256,1)
    orig = cv2.imread(image_path)
    orig_resized = cv2.resize(orig, (256, 256))

    # --- soft overlay: blurred mask + heatmap ---
    mask_gray = pred_mask.squeeze().astype(np.uint8)
    soft_mask = cv2.GaussianBlur(mask_gray, (11, 11), 0)
    soft_mask = soft_mask.astype(np.float32) / 255.0

    heatmap = cv2.applyColorMap(mask_gray, cv2.COLORMAP_MAGMA)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_f = orig_resized.astype(np.float32) / 255.0
    heat_f = heatmap.astype(np.float32) / 255.0
    alpha = (soft_mask * 0.6)[..., None]  # up to 60% tint

    blended = orig_f * (1 - alpha) + heat_f * alpha
    overlay_rgb = (blended * 255).astype(np.uint8)

    # subtle white contour on boundary
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, -1, (255, 255, 255), 1)

    return orig_resized, mask_gray, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), runtime

# ----------------- SESSION STATE -----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------- HEADER -----------------
st.markdown('<a name="analyzer"></a>', unsafe_allow_html=True)
st.title("Brain Tumor Segmentation")
st.markdown(
    '<div class="caption">Upload an MRI brain scan, add patient details, and run automatic tumor segmentation with slice-wise tumor area estimation.</div>',
    unsafe_allow_html=True
)

# ----------------- PATIENT DETAILS CARD -----------------
st.markdown('<div class="patient-card">', unsafe_allow_html=True)
st.markdown("#### Patient details", unsafe_allow_html=True)

pd_col1, pd_col2, pd_col3 = st.columns([1.2, 1.0, 1.2])
with pd_col1:
    patient_name = st.text_input("Patient name", placeholder="e.g., John Doe")
    patient_id = st.text_input("Patient ID / MRN", placeholder="e.g., MRN-001")
with pd_col2:
    patient_age = st.text_input("Age", placeholder="e.g., 54")
    patient_gender = st.selectbox("Gender", ["-", "Male", "Female", "Other"], index=0)
with pd_col3:
    patient_notes = st.text_area(
        "Clinical notes",
        placeholder="Optional clinical context (symptoms, findings)...",
        height=80
    )

st.markdown("#### Upload MRI", unsafe_allow_html=True)

# ---- small centered uploader ----
st.markdown('<div class="upload-wrapper">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,       # still supports multiple, just not written anywhere
    key="multiupload",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Analyze button (prevents auto-processing on every rerun)
analyze_col = st.columns([1, 1, 1])[1]
with analyze_col:
    analyze_clicked = st.button("Analyze MRI", key="analyze_button", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)  # end patient-card

current_patient = {
    "name": (patient_name or "").strip(),
    "id": (patient_id or "").strip(),
    "age": (patient_age or "").strip(),
    "gender": patient_gender if patient_gender != "-" else "",
    "notes": (patient_notes or "").strip()
}

# ----------------- PROCESS WHEN ANALYZE CLICKED -----------------
if analyze_clicked and uploaded_files:
    for uploaded_file in uploaded_files:
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
            caption="Tumor highlight (soft overlay)",
            use_container_width=True
        )

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

        # --- avoid duplicate entries in history for same file + patient ---
        exists = any(
            (h["name"] == uploaded_file.name) and
            (h.get("patient", {}).get("id") == current_patient.get("id")) and
            (h.get("patient", {}).get("name") == current_patient.get("name"))
            for h in st.session_state.history
        )
        if not exists:
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

# ----------------- SESSION HISTORY -----------------
st.markdown('<a name="session-history"></a>', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("## Session history")

    for item in reversed(st.session_state.history):
        patient = item.get("patient", {}) or {}
        p_name = patient.get("name") or "Unknown patient"
        p_id = patient.get("id") or "N/A"
        p_age = patient.get("age") or "N/A"
        p_gender = patient.get("gender") or "-"
        p_notes = patient.get("notes") or ""

        st.markdown('<div class="history-card">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="history-header">
              <div>
                <div class="hist-patient-name">{p_name}</div>
                <div class="hist-patient-meta">
                    ID: {p_id} · Age: {p_age} · Gender: {p_gender}
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
    st.info("No history yet. Click **Analyze MRI** after uploading to see results.")
