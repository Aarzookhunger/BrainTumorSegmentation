import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tempfile
import time

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.markdown("""
<style>
    body, .stApp {
        background-color: #f8fafc;
    }
    .main .block-container {
        background-color: white;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        max-width: 900px;
        margin: 2rem auto;
    }
    h1 {
        color: #1f2937;
        text-align: center;
        font-size: 2.2rem;
        margin-bottom: 1rem;
    }
    .caption {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .stDownloadButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease-in;
    }
    .stDownloadButton button:hover {
        background-color: #1d4ed8;
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
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
    overlay = orig_resized.copy()
    overlay[pred_mask.squeeze() == 255] = [0, 0, 255]
    return orig_resized, pred_mask.squeeze(), overlay, runtime

st.title("Brain Tumor Segmentation")
st.markdown('<p class="caption">Upload an MRI scan to automatically detect and highlight tumor regions.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing MRI scan..."):
        orig, mask, overlay, runtime = predict_tumor(tmp_path)

    st.success(f"Processing complete in {runtime:.2f}s")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    with col2:
        st.image(mask, caption="Predicted Mask", use_container_width=True, clamp=True)
    with col3:
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Tumor Highlight", use_container_width=True)

    tumor_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    tumor_pct = (tumor_pixels / total_pixels) * 100

    st.metric("Detected Tumor Area", f"{tumor_pct:.2f}%")

    mask_img = Image.fromarray(mask)
    overlay_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    buf_mask = io.BytesIO()
    buf_overlay = io.BytesIO()
    mask_img.save(buf_mask, format="PNG")
    overlay_img.save(buf_overlay, format="PNG")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Segmentation Mask", buf_mask.getvalue(), f"mask_{uploaded_file.name}", "image/png")
    with c2:
        st.download_button("Download Tumor Overlay", buf_overlay.getvalue(), f"overlay_{uploaded_file.name}", "image/png")

else:
    st.info("Please upload an image to begin analysis.")
