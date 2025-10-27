# Brain Tumor Segmentation

A simple, modern web application for **automatic detection and highlighting of brain tumor regions from MRI scans.**  
Built with [Streamlit](https://streamlit.io/), this tool allows users to upload one or more MRI images and returns visual overlays marking the detected tumor regions.

---

## 🚀 Live Demo

Access the app at:  
**[_https://braintumorsegmentation-u7bgpkqnz29weqnpfzfka9.streamlit.app/_](#)**

---

## Features

- **Batch Upload**: Upload one or more MRI scan images at once.
- **Instant Analysis**: Tumor regions are detected and highlighted within seconds.
- **Result Downloads**: Download both the binary tumor mask and the colored overlay for each image.
- **Session History**: Quickly review results from all images processed in your session.
- **No Setup Needed**: Works out of the box for doctors, researchers, students, or anyone needing quick brain tumor visualization.

---

## How To Use

1. **Visit the app link above** and wait for it to load.
2. **Upload your MRI scan(s)** (JPG/PNG).
3. The app will show:
    - The original image
    - Tumor segmentation mask
    - Overlay with tumor highlighted in blue
    - [Optional] Tumor area percentage
4. **Download your results** for each image.
5. Scroll down to view the **session history gallery** of all scans processed in your session.

---

## Technical Aspect

- The app uses a deep learning model for segmentation (EfficientNet-based, trained on open datasets).
- All analysis is performed securely on the server; **images are never stored** beyond the browser session.

---

## Deployment & Usage

- No installation or registration required.  
- Works directly from any web browser.

---
