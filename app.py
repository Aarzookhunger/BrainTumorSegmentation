import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image
import io
import time
from base64 import b64encode

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.markdown("""
<style>
body, .stApp {
    background: #141927;
}
.main .block-container {
    background: #191c27;
    border-radius: 14px;
    box-shadow: 0 3px 18px rgba(0,0,0,0.15);
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

/* Top Navbar */
.top-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(12px);
    border-radius: 0 0 14px 14px;
    border-bottom: 1px solid #1f2937;
    padding: 0.7rem 1.4rem;
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
    font-size: 1.1rem;
}
.nav-right {
    display: flex;
    gap: 1rem;
    font-size: 0.9rem;
}
.nav-link {
    color: #9ca3af;
    text-decoration: none;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    border: 1px solid transparent;
}
.nav-link:hover {
    color: #e5e7eb;
    border-color: #374151;
    background: #111827;
}

h1 {
    color: #e0e0e0;
    text-align: center;
    font-size: 2.2rem;
    margin-bottom: 1rem;
}
.caption { 
    color: #b0b5be; 
    text-align: center;
    font-size: 1.05rem;
    margin-bottom: 2.0rem;
}
.stDownloadButton button {
    background-color: #3b82f6;
    color: #eee;
    border-radius: 7px;
    border: none;
    padding: 0.55rem 1.3rem;
    font-weight: 500;
    margin-top: 0.4rem;
}
.stDownloadButton button:hover {
    background: linear-gradient(90deg,#486eb7,#3493da);
    color: #fff;
}
hr {border-top: 2px solid #2e3355;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="top-nav">
  <div class="nav-left">
    <span class="nav-logo">🧠</span>
    <span class="nav-title">Brain Tumor Studio</span>
  </div>
  <div class="nav-right">
    <a href="#analyzer" class="nav-link">Analyzer</a>
    <a href="#session-history" class="nav-link">Session history</a>
  </div>
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


def predict_tumor(image_path):
    pre_img = preprocess_image(image_path)
    input_img = np.expand_dims(pre_img, axis=0)
    start_time = time.time()
    pred_mask = model.predict(input_img, verbose=0)[0]
    runtime = time.time() - start_time
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    orig = cv2.imread(image_path)
    orig_resized = cv2.resize(orig, (256, 256))

    # --- Improved blending for overlay ---
    color_layer = np.zeros_like(orig_resized)
    # light cyan-ish tint for tumor region
    color_layer[:] = (15, 159, 253)  # BGR
    blended_full = cv2.addWeighted(orig_resized, 0.65, color_layer, 0.35, 0)

    overlay = orig_resized.copy()
    mask_bool = pred_mask.squeeze() == 255
    overlay[mask_bool] = blended_full[mask_bool]

    # Optional: thin white contour for tumor boundary
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return orig_resized, pred_mask.squeeze(), overlay, runtime


def image_with_tooltip(container, img_array, caption, tooltip):
    """Render an image with HTML-based tooltip using <img title="...">."""
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = b64encode(buf.getvalue()).decode()
    html = f"""
    <div style="display:inline-block; width:100%;" title="{tooltip}">
      <img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;" />
      <div style="text-align:center;font-size:0.80rem;color:#9ca3af;margin-top:0.25rem;">
        {caption}
      </div>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)


if "history" not in st.session_state:
    st.session_state.history = []


st.markdown('<a name="analyzer"></a>', unsafe_allow_html=True)
st.title("Brain Tumor Segmentation")
st.markdown(
    '<div class="caption">Upload one or more MRI scans to analyze tumor regions.<br>'
    'All results are kept in your session history below.</div>',
    unsafe_allow_html=True
)

# ----------------- Patient Details -----------------
with st.expander("Patient details (applied to this upload batch)", expanded=True):
    pd_col1, pd_col2 = st.columns(2)
    patient_name = pd_col1.text_input("Patient name", placeholder="e.g., John Doe")
    patient_id = pd_col1.text_input("Patient ID / MRN", placeholder="e.g., MRN-001")
    patient_age = pd_col2.text_input("Age", placeholder="e.g., 54")
    patient_notes = pd_col2.text_area("Clinical notes / remarks", placeholder="Optional clinical context...")

current_patient = {
    "name": patient_name.strip() if patient_name else "",
    "id": patient_id.strip() if patient_id else "",
    "age": patient_age.strip() if patient_age else "",
    "notes": patient_notes.strip() if patient_notes else ""
}

# ----------------- Uploader -----------------
uploaded_files = st.file_uploader(
    "Upload MRI",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="multiupload"
)

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

        st.markdown(f'<hr>', unsafe_allow_html=True)
        cols = st.columns([2, 2, 2, 2])

        # Display original
        cols[0].image(
            cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
            caption="Original MRI slice",
            use_container_width=True
        )

        # Display mask
        cols[1].image(
            mask,
            caption="Predicted Mask",
            use_container_width=True,
            clamp=True
        )

        # Tooltip text with patient + tumor info
        tooltip_text = (
            f"Patient: {current_patient.get('name') or 'N/A'} | "
            f"ID: {current_patient.get('id') or 'N/A'} | "
            f"Age: {current_patient.get('age') or 'N/A'} | "
            f"Tumor area: {tumor_pct:.2f}%"
        )

        # Tumor highlight with hover tooltip
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image_with_tooltip(
            cols[2],
            overlay_rgb,
            "Tumor Highlight (hover for patient info)",
            tooltip_text
        )

        cols[3].metric(
            "Tumor Area (slice)",
            f"{tumor_pct:.2f}%",
            help="Approximate tumor-covered area as percentage of this MRI slice."
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
        c1.download_button("Download Mask", buf_mask.getvalue(),
                           f"mask_{uploaded_file.name}", "image/png")
        c2.download_button("Download Overlay", buf_overlay.getvalue(),
                           f"overlay_{uploaded_file.name}", "image/png")

        # Save in session history (including patient details)
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

# ----------------- Session History -----------------
st.markdown('<a name="session-history"></a>', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown(
        '<h3 style="color:#fff; margin-top:2rem;">Session History</h3>',
        unsafe_allow_html=True
    )
    for i, item in enumerate(st.session_state.history[::-1]):
        patient = item.get("patient", {}) or {}
        p_name = patient.get("name") or "Unknown"
        p_id = patient.get("id") or "N/A"
        p_age = patient.get("age") or "N/A"

        tooltip_text_hist = (
            f"Patient: {p_name} | ID: {p_id} | Age: {p_age} | "
            f"Tumor area: {item.get('tumor_pct', 'N/A')}"
        )

        ch = st.columns([2, 2, 2, 1.2])

        ch[0].image(item["original"],
                    caption=f"Original ({item['name']})",
                    width=120)

        ch[1].image(item["mask"], caption="Mask", width=120, clamp=True)

        # Overlay with tooltip in history
        image_with_tooltip(
            ch[2],
            item["overlay"],
            "Overlay (hover: patient info)",
            tooltip_text_hist
        )

        ch[3].markdown(
            f"""
            <div style="color:#5eead4;font-weight:500;font-size:0.95rem;">
                {item['tumor_pct']}
            </div>
            <div style="color:#9ca3af;font-size:0.75rem;">
                {p_name} ({p_id})
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("No history yet. Process images to view session history.")
