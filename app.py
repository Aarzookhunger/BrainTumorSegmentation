import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tempfile
from PIL import Image
import io
import time

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ===================== CUSTOM METRICS & LOSSES (USED DURING TRAINING) =====================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def boundary_loss(y_true, y_pred):
    kernel = tf.constant([[0., 1., 0.],
                          [1., -4., 1.],
                          [0., 1., 0.]], dtype=tf.float32)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    edge_true = tf.nn.conv2d(y_true, kernel, strides=[1, 1, 1, 1], padding='SAME')
    edge_pred = tf.nn.conv2d(y_pred, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.reduce_mean(tf.abs(edge_true - edge_pred))

def hybrid_loss(y_true, y_pred, w_bce=0.5, w_dice=0.4, w_bound=0.1):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dloss = dice_loss(y_true, y_pred)
    bl = boundary_loss(y_true, y_pred)
    return w_bce * bce + w_dice * dloss + w_bound * bl

def hybrid_loss_wrapped(y_true, y_pred):
    # Wrapper used when saving/loading the model with custom_objects
    return hybrid_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)

def precision_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    return tp / (tp + fp + smooth)

def recall_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return tp / (tp + fn + smooth)

def specificity_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    return tn / (tn + fp + smooth)

# ===================== STYLING =====================
st.markdown("""
<style>
body, .stApp {
    background: #020617;
    color: #e5e7eb !important;
}

.main .block-container {
    background: #020617;
    border-radius: 0;
    box-shadow: none;
    padding-top: 1rem;
    padding-bottom: 2rem;
    border: none;
}

/* Titles */
h1, h2, h3, h4, h5, h6 {
    color: #e5e7eb !important;
}
h1 {
    font-size: 2.4rem !important;
    margin-bottom: 0.4rem;
    font-weight: 800 !important;
}
.caption {
    font-size: 0.95rem;
    margin-bottom: 1rem;
    color: #9ca3af !important;
}
label, p, span, div {
    color: #e5e7eb;
}

/* Patient details card */
.patient-card {
    background: #020617;
    border-radius: 12px;
    border: 1px solid #1f2937;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    border: 1px solid #4b5563 !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    font-size: 0.85rem !important;
    color: #e5e7eb !important;
}

/* Buttons */
.stButton button {
    border-radius: 999px !important;
    padding: 0.4rem 1.2rem !important;
    font-weight: 600 !important;
    border: 1px solid #4b5563 !important;
    background: #111827 !important;
    color: #e5e7eb !important;
}
.stButton button:hover {
    background: #1f2937 !important;
}

/* Analyze button: bluish */
.analyze-btn button {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
}

/* Centered small upload block */
.center-upload {
    max-width: 260px;
    margin: 0 auto;
}
.center-upload [data-testid="stFileUploaderDropzone"] {
    border-radius: 999px !important;
    padding: 0.2rem 0.5rem !important;
    border: 1px solid #4b5563 !important;
    background-color: #020617 !important;
    min-height: 40px !important;
}
.center-upload [data-testid="stFileUploaderDropzone"] > div {
    justify-content: center !important;
}

/* Hide drag/drop + 200MB text, keep only Browse button */
.center-upload [data-testid="stFileUploaderDropzone"] > div:first-child {
    display: none !important;
}
.center-upload [data-testid="stFileUploader"] button * {
    color: #e5e7eb !important;
    font-size: 0.85rem !important;
}

/* Session history cards */
.history-card {
    background: #020617;
    border-radius: 12px;
    border: 1px solid #1f2937;
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
    font-size: 1.2rem;
    color: #e5e7eb;
}
.hist-patient-meta {
    font-size: 1.0rem;
    color: #cbd5f5;
}
.hist-tumor {
    font-size: 1.05rem;
    font-weight: 800;
    color: #22c55e;
}

/* Tumor box (current result) */
.tumor-box-label {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
    color: #cbd5f5;
}
.tumor-box-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #38bdf8;
}
.tumor-extra {
    font-size: 0.90rem;
    color: #9ca3af;
}

/* Misc */
hr {border-top: 1px solid #1f2937;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===================== MODEL =====================
@st.cache_resource
def load_segmentation_model():
    # custom_objects are registered so the model can be loaded even if it was saved with these losses/metrics
    return load_model(
        "final_model.keras",
        compile=False,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "boundary_loss": boundary_loss,
            "hybrid_loss": hybrid_loss,
            "hybrid_loss_wrapped": hybrid_loss_wrapped,
            "iou_metric": iou_metric,
            "precision_metric": precision_metric,
            "recall_metric": recall_metric,
            "specificity_metric": specificity_metric,
        }
    )

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

# ===================== SESSION STATE =====================
if "history" not in st.session_state:
    st.session_state.history = []

# ===================== HEADER =====================
st.title("🧠 Brain Tumor Segmentation")
st.markdown(
    '<div class="caption">Upload an MRI brain scan, add patient details, then run tumor segmentation to estimate tumor area on the slice.</div>',
    unsafe_allow_html=True
)

# ===================== PATIENT DETAILS =====================
st.markdown('<div class="patient-card">', unsafe_allow_html=True)
st.markdown("#### Patient details", unsafe_allow_html=True)

pd_left, pd_right = st.columns(2)
with pd_left:
    patient_name = st.text_input("Patient name")
    patient_id = st.text_input("Patient ID / MRN")
with pd_right:
    patient_age = st.text_input("Age")
    patient_gender = st.selectbox("Gender", ["-", "Male", "Female", "Other"], index=0)

patient_notes = st.text_area("Clinical notes", height=110)

st.markdown("#### Upload MRI scan of the patient", unsafe_allow_html=True)

# ---- Centered small upload + centered analyze ----
sp1, mid, sp2 = st.columns([3, 2, 3])
with mid:
    st.markdown('<div class="center-upload">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multiupload",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="analyze-btn" style="margin-top:0.6rem; text-align:center;">', unsafe_allow_html=True)
    analyze_clicked = st.button("Analyze MRI")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end patient-card

current_patient = {
    "name": (patient_name or "").strip(),
    "id": (patient_id or "").strip(),
    "age": (patient_age or "").strip(),
    "gender": patient_gender if patient_gender != "-" else "",
    "notes": (patient_notes or "").strip()
}

# ===================== RUN ANALYSIS WHEN CLICKED =====================
if analyze_clicked and uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            orig, mask, overlay, runtime = predict_tumor(tmp_path)

        tumor_pixels = int(np.sum(mask == 255))
        total_pixels = int(mask.size)
        tumor_pct = (tumor_pixels / total_pixels) * 100

        # -------- Row with patient + numeric details BEFORE images --------
        info_left, info_right = st.columns(2)
        with info_left:
            st.markdown(
                f"**Patient:** {current_patient.get('name') or 'Not specified'}  \n"
                f"**ID:** {current_patient.get('id') or '-'}  \n"
                f"**Age:** {current_patient.get('age') or '-'}  \n"
                f"**Gender:** {current_patient.get('gender') or '-'}  \n"
                f"**Notes:** {current_patient.get('notes') or '-'}"
            )
        with info_right:
            st.markdown(
                f"""
                <div class="tumor-box-label">Tumor area (this slice)</div>
                <div class="tumor-box-value">{tumor_pct:.2f}%</div>
                <div class="tumor-extra">
                    Pixels in tumor: {tumor_pixels}<br>
                    Slice size: {mask.shape[0]} × {mask.shape[1]}<br>
                    Runtime: {runtime:.2f}s<br><br>
                    <b>Model test Dice (DSC):</b> 95.7%
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -------- images + aligned download buttons --------
        buf_mask = io.BytesIO()
        Image.fromarray(mask).save(buf_mask, format="PNG")

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        buf_overlay = io.BytesIO()
        Image.fromarray(overlay_rgb).save(buf_overlay, format="PNG")

        img_cols = st.columns(3)

        with img_cols[0]:
            st.image(
                cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
                caption=f"Original MRI slice ({uploaded_file.name})",
                use_container_width=True
            )

        with img_cols[1]:
            st.image(
                mask,
                caption="Predicted mask",
                use_container_width=True,
                clamp=True
            )
            st.download_button(
                "Download mask",
                buf_mask.getvalue(),
                f"mask_{uploaded_file.name}",
                "image/png",
                use_container_width=True
            )

        with img_cols[2]:
            st.image(
                overlay_rgb,
                caption="Tumor highlight",
                use_container_width=True
            )
            st.download_button(
                "Download overlay",
                buf_overlay.getvalue(),
                f"overlay_{uploaded_file.name}",
                "image/png",
                use_container_width=True
            )

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
                "runtime": runtime,
                "tumor_pixels": tumor_pixels,
                "total_pixels": total_pixels,
                "patient": current_patient.copy()
            })

st.markdown("<hr>", unsafe_allow_html=True)

# ===================== SESSION HISTORY =====================
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

        hcols = st.columns(3)
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
