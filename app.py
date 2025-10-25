import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tempfile

# ------------------ Page Config ------------------
st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="wide")

# ------------------ Load Model ------------------
@st.cache_resource
def load_segmentation_model():
    model = load_model("final_model.keras", compile=False)
    return model

model = load_segmentation_model()

# ------------------ Preprocessing Function ------------------
def preprocess_image(image_path, size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE().apply(l)
    lab = cv2.merge((clahe, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

# ------------------ Prediction Function ------------------
def predict_tumor(image_path):
    pre_img = preprocess_image(image_path)
    input_img = np.expand_dims(pre_img, axis=0)
    pred_mask = model.predict(input_img, verbose=0)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    orig = cv2.imread(image_path)
    orig_resized = cv2.resize(orig, (256, 256))
    overlay = orig_resized.copy()
    overlay[pred_mask.squeeze() == 255] = [0, 0, 255]
    
    return orig_resized, pred_mask.squeeze(), overlay

# ------------------ UI ------------------
st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload an MRI image to detect and visualize tumor regions using deep learning.")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.success("Image uploaded successfully!")
    
    # Run prediction
    with st.spinner("Running segmentation..."):
        orig, mask, overlay = predict_tumor(tmp_path)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original MRI")
        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("Predicted Mask")
        st.image(mask, use_container_width=True, clamp=True)
    
    with col3:
        st.subheader("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    st.success("✅ Segmentation complete!")