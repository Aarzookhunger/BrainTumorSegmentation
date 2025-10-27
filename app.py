import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image
import io
import time

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
h1 {
    color: #e0e0e0;
    text-align: center;
    font-size: 2.2rem;
    margin-bottom: 1rem;
}
.caption { color: #b0b5be; text-align: center;
    font-size: 1.05rem;
    margin-bottom: 2.0rem;}
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
    overlay[pred_mask.squeeze() == 255] = [15, 159, 253]
    return orig_resized, pred_mask.squeeze(), overlay, runtime

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Brain Tumor Segmentation")
st.markdown('<div class="caption">Upload one or more MRI scans to analyze for tumor regions.<br>All results are kept in your session history below.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload MRI Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multiupload")

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
        cols = st.columns([2,2,2,2])
        cols[0].image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        cols[1].image(mask, caption="Predicted Mask", use_container_width=True, clamp=True)
        cols[2].image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Tumor Highlight", use_container_width=True)
        cols[3].metric("Tumor Area", f"{tumor_pct:.2f}%")
        st.info(f"Processing time: {runtime:.2f}s", icon="ℹ️")

        mask_img = Image.fromarray(mask)
        overlay_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        buf_mask = io.BytesIO()
        buf_overlay = io.BytesIO()
        mask_img.save(buf_mask, format="PNG")
        overlay_img.save(buf_overlay, format="PNG")
        c1, c2 = st.columns(2)
        c1.download_button("Download Mask", buf_mask.getvalue(), f"mask_{uploaded_file.name}", "image/png")
        c2.download_button("Download Overlay", buf_overlay.getvalue(), f"overlay_{uploaded_file.name}", "image/png")

        st.session_state.history.append({
            "name": uploaded_file.name,
            "original": cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
            "mask": mask.copy(),
            "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            "tumor_pct": f"{tumor_pct:.2f}%"
        })

st.markdown("<hr>", unsafe_allow_html=True)

if st.session_state.history:
    st.markdown('<h3 style="color:#fff; margin-top:2rem;">Session History</h3>', unsafe_allow_html=True)
    for i, item in enumerate(st.session_state.history[::-1]):
        ch = st.columns([2,2,2,1.1])
        ch[0].image(item["original"], caption=f"Original ({item['name']})", width=120)
        ch[1].image(item["mask"], caption="Mask", width=120, clamp=True)
        ch[2].image(item["overlay"], caption="Overlay", width=120)
        ch[3].markdown(f"<span style='color:#5eead4;font-weight:500'>{item['tumor_pct']}</span>", unsafe_allow_html=True)
else:
    st.info("No history yet. Process images to view session history.")
