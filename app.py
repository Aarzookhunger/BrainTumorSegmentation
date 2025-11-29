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
body, .stApp {
    background: #ddefff;  /* light medical blue */
    color: #000000 !important;
}

/* Main card */
.main .block-container {
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.10);
    padding-top: 1rem;
    padding-bottom: 2rem;
    border: 1px solid #e5e7eb;
}

/* Titles and general text */
h1, h2, h3, h4, h5, h6, label, p, span, div {
    color: #000000 !important;
}
h1 {
    font-size: 1.9rem !important;
}
.caption {
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

/* Patient card */
.patient-card {
    background: #f5f7ff;
    border-radius: 12px;
    border: 1px solid #c8d6f2;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 8px !important;
    border: 1px solid #9ca3af !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    font-size: 0.85rem !important;
}

/* Upload – make it look like a small button, not a long pill */
.upload-inline {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.5rem;
    color: #ffffff
}

.upload-inline [data-testid="stFileUploader"] {
    max-width: 100px;
}

/* hide the big drop area and leave only the button */
.upload-inline [data-testid="stFileUploaderDropzone"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
.upload-inline [data-testid="stFileUploaderDropzone"] > div {
    padding: 0 !important;
    margin: 0 !important;
    justify-content: center !important;
}
.upload-inline [data-testid="stFileUploaderDropzone"] span {
    display: none !important;  /* no "Drag and drop..." etc. */
}

/* Buttons */
.stButton button {
    border-radius: 999px ;
    padding: 0.35rem 1.1rem ;
}
.analyze-btn button {
    background-color: #000000 ;
    color: #ffffff ;
}

/* Download buttons */
.stDownloadButton button {
    background-color: #2563eb;
    color: #ffffff;
    border-radius: 8px;
    border: none;
    padding: 0.45rem 1.1rem;
    font-weight: 500;
    margin-top: 0.3rem;
}
.stDownloadButton button:hover {
    background: #1d4ed8;
}

/* Session history cards */
.history-card {
    background: #f5f7ff;
    border-radius: 12px;
    border: 1px solid #d1d5db;
    padding: 0.9rem 1.0rem;
    margin-bottom: 1rem;
}
.history-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.4rem;
}
.hist-patient-name {
    font-weight: 800;
    font-size: 1.25rem;    /* bigger */
}
.hist-patient-meta {
    font-size: 1.0rem;     /* bigger */
}
.hist-tumor {
    font-size: 1.05rem;
    font-weight: 800;
    color: #065f46;
}

/* Tumor metric custom text */
.tumor-box-label {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.tumor-box-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #2563eb;
}

/* Misc */
hr {border-top: 1px solid #e5e7eb;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
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

    # Soft overlay
    mask_gray = pred_mask.squeeze().astype(np.uint8)
    soft_mask = cv2.GaussianBlur(mask_gray, (11, 11), 0)
    soft_mask = soft_mask.astype(np.float32) / 255.0

    heatmap = cv2.applyColorMap(mask_gray, cv2.COLORMAP_MAGMA)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_f = orig_resized.astype(np.float32) / 255.0
    heat_f = heatmap.astype(np.float32) / 255.0
    alpha = (soft_mask * 0.6)[..., None]

    blended = orig_f * (1 - alpha) + heat_f * alpha
    overlay_rgb = (blended * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, -1, (255, 255, 255), 1)

    return orig_resized, mask_gray, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), runtime

# ----------------- SESSION STATE -----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------- HEADER -----------------
st.title("Brain Tumor Segmentation")
st.markdown(
    '<div class="caption">Upload an MRI brain scan, add patient details, then run tumor segmentation to estimate tumor area on the slice.</div>',
    unsafe_allow_html=True
)

# ----------------- PATIENT DETAILS -----------------
st.markdown('<div class="patient-card">', unsafe_allow_html=True)
st.markdown("#### Patient details", unsafe_allow_html=True)

pd_col1, pd_col2, pd_col3 = st.columns([1.2, 1.0, 1.2])
with pd_col1:
    patient_name = st.text_input("Patient name")
    patient_id = st.text_input("Patient ID")
with pd_col2:
    patient_age = st.text_input("Age", placeholder="e.g., 54")
    patient_notes = st.text_area("Clinical notes", placeholder="Optional remarks...", height=80)

st.markdown("#### MRI controls", unsafe_allow_html=True)

# ---- Upload + Analyze buttons in one centered row ----
c_left, c_mid, c_right = st.columns([1, 2, 1])
with c_mid:
    st.markdown('<div class="upload-inline">', unsafe_allow_html=True)

    # Upload MRI (small)
    upload_col, analyze_col = st.columns([1, 1])

    with upload_col:
        uploaded_files = st.file_uploader(
            "",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    with analyze_col:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze MRI")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end patient card

current_patient = {
    "name": (patient_name or "").strip(),
    "id": (patient_id or "").strip(),
    "age": (patient_age or "").strip(),
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
            caption="Tumor highlight",
            use_container_width=True
        )

        with cols[3]:
            st.markdown(
                f"""
                <div class="tumor-box-label">Tumor area (this slice)</div>
                <div class="tumor-box-value">{tumor_pct:.2f}%</div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            st.markdown(
                f"**Patient:** {current_patient.get('name') or 'Not specified'}  \n"
                f"**ID:** {current_patient.get('id') or '-'}  \n"
                f"**Age:** {current_patient.get('age') or '-'}",
            )

        st.info(f"Processing time: {runtime:.2f}s", icon="ℹ️")

        # Downloads
        mask_img = Image.fromarray(mask)
        buf_mask = io.BytesIO()
        mask_img.save(buf_mask, format="PNG")

        overlay_img = Image.fromarray(overlay_rgb)
        buf_overlay = io.BytesIO()
        overlay_img.save(buf_overlay, format="PNG")

        d1, d2 = st.columns(2)
        d1.download_button("Download mask", buf_mask.getvalue(),
                           f"mask_{uploaded_file.name}", "image/png")
        d2.download_button("Download overlay", buf_overlay.getvalue(),
                           f"overlay_{uploaded_file.name}", "image/png")

        # Avoid duplicates in history
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
                    ID: {p_id} · Age: {p_age}
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

        hcols = st.columns([2, 2, 2])
        hcols[0].image(item["original"],
                       caption=f"Original ({item['name']})",
                       use_container_width=True)
        hcols[1].image(item["mask"],
                       caption="Mask",
                       use_container_width=True,
                       clamp=True)
        hcols[2].image(item["overlay"],
                       caption="Overlay",
                       use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No history yet. Upload an MRI and click **Analyze MRI** to generate results.")


