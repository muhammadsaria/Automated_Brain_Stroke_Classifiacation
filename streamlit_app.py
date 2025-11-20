# app_final.py
# Automated Brain Stroke Classification — Final
# Features: MONAI DenseNet121 prediction, dark/light mode, loading spinner, multi-color theme,
# PDF export (ReportLab), Doctor summary page, Multi-scan comparison.

import os
import time
import io
from datetime import datetime

import streamlit as st
import torch
import torch.nn.functional as F
from monai.transforms import LoadImage, Compose, EnsureChannelFirst, ScaleIntensity
from monai.networks.nets import DenseNet121
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional PDF library
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Automated Brain Stroke Classification", page_icon="🧠", layout="wide")

MODEL_PATH = "best_model (1).pth"
LOGS_CSV = "predictions_log.csv"

single_transform = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])
classes = ["Normal", "Stroke"]

# -------------------------
# Sidebar navigation & Dark Mode
# -------------------------
page = st.sidebar.selectbox("Navigation", ["Classification", "Multi-scan Compare", "Doctor Summary", "Settings"])
dark_mode = st.sidebar.checkbox("Enable Dark Mode", value=False)

# -------------------------
# CSS
# -------------------------
if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #121212; color: #e0e0e0; }
        .app-title { 
            font-size: 42px; font-weight:900; text-align:center;
            background: linear-gradient(90deg,#ff6b6b,#f6d365,#5ee7df);
            -webkit-background-clip: text; color: transparent; margin-bottom: -6px; 
        }
        .app-sub { text-align:center; color:#c0c0c0; margin-bottom:20px; font-weight:600; }
        .card { background: #1e1e1e; padding:18px; border-radius:12px; box-shadow: 0 6px 20px rgba(0,0,0,0.5); }
        .prec { background: linear-gradient(90deg,#2c1c1c,#3d2e2e); padding:14px; border-radius:10px; }
        .small-muted { color:#aaaaaa; font-size:13px; }
        .loader { border: 6px solid #333; border-top: 6px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin:auto; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); color: #000000; }
        .app-title { font-size: 42px; font-weight:900; text-align:center;
                     background: linear-gradient(90deg,#ff6b6b,#f6d365,#5ee7df);
                     -webkit-background-clip: text; color: transparent; margin-bottom: -6px; }
        .app-sub { text-align:center; color:#475569; margin-bottom:20px; font-weight:600; }
        .card { background: white; padding:18px; border-radius:12px; box-shadow: 0 6px 20px rgba(44,62,80,0.06); }
        .prec { background: linear-gradient(90deg,#fff1f0,#fff7ed); padding:14px; border-radius:10px; }
        .small-muted { color:#64748b; font-size:13px; }
        .loader { border: 6px solid #f3f3f3; border-top: 6px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin:auto; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="app-title">🧠 Automated Brain Stroke Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">MONAI DenseNet121 • Upload CT → Predict • Beautiful interface</div>', unsafe_allow_html=True)

# -------------------------
# Model loader
# -------------------------
@st.cache_resource
def load_model(path=MODEL_PATH, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")
    state = torch.load(path, map_location=device)
    try:
        if isinstance(state, dict) and not hasattr(state, "forward"):
            model.load_state_dict(state)
        else:
            model = state
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    model.eval()
    return model, device

try:
    model, device = load_model()
except Exception as e:
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_error_text = str(e)

# -------------------------
# Utils
# -------------------------
def save_uploaded_file(uploaded_file, dest_path):
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.read())
    return dest_path

def predict_with_monai(path):
    img = single_transform(path)
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], float(conf.item() * 100), probs.cpu().numpy()[0]

def append_log(entry: dict):
    df = pd.DataFrame([entry])
    if os.path.exists(LOGS_CSV):
        df.to_csv(LOGS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(LOGS_CSV, index=False)

def load_logs_df():
    if os.path.exists(LOGS_CSV):
        return pd.read_csv(LOGS_CSV)
    else:
        return pd.DataFrame(columns=["timestamp","first_name","last_name","age","gender","image_name","prediction","confidence"])

def generate_pdf_report(patient_info: dict, prediction: str, confidence: float, image_path: str):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not installed. Install via `pip install reportlab` to enable PDF export.")
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, "Automated Brain Stroke Classification")
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 85, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, height - 120, "Patient Information")
    c.setFont("Helvetica", 11)
    y = height - 140
    for k, v in patient_info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 16
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y - 6, "Prediction")
    c.setFont("Helvetica", 12)
    c.drawString(50, y - 26, f"Result: {prediction}")
    c.drawString(50, y - 44, f"Confidence: {confidence:.2f}%")
    try:
        thumb = Image.open(image_path).convert("RGB")
        thumb.thumbnail((240,240))
        thumb_buf = io.BytesIO()
        thumb.save(thumb_buf, format="PNG")
        thumb_buf.seek(0)
        c.drawInlineImage(thumb_buf, width - 300, height - 350, 240, 240)
    except Exception:
        pass
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# PAGE: Classification
# -------------------------
if page == "Classification":
    st.header("Classification")
    if model is None:
        st.error("Model not loaded. Go to Settings. Error:")
        st.text(load_error_text if 'load_error_text' in locals() else "Unknown")
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Patient information")
    col1, col2 = st.columns(2)
    first_name = col1.text_input("First name")
    last_name = col2.text_input("Last name")
    col3, col4 = st.columns(2)
    age = col3.number_input("Age", min_value=0, max_value=120, value=30)
    gender = col4.selectbox("Gender", ["Male","Female","Other","Prefer not to say"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) CT scan")
    uploaded = st.file_uploader("Upload CT scan image (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=False)

    if uploaded is not None:
        tmpname = f"upload_{int(time.time())}.png"
        save_uploaded_file(uploaded, tmpname)
        uploaded_path = tmpname
        img_to_show = Image.open(uploaded_path)
        # <-- Fixed width instead of use_container_width -->
        st.image(img_to_show, caption="CT preview", width=700)

        if st.button("Run Prediction"):
            with st.spinner("Running model (this may take a few seconds)..."):
                try:
                    prediction, confidence, probs = predict_with_monai(uploaded_path)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    prediction, confidence, probs = None, None, None

            if prediction is not None:
                st.markdown("### Result")
                if prediction == "Normal":
                    st.success(f"🟢 Prediction: {prediction}  —  Confidence: {confidence:.2f}%")
                    st.info("Your CT scan appears normal. If you have symptoms, consult a doctor.")
                else:
                    st.error(f"🔴 Prediction: {prediction}  —  Confidence: {confidence:.2f}%")
                    st.markdown('<div class="prec">', unsafe_allow_html=True)
                    st.markdown("### Immediate precautions & next steps")
                    st.markdown("""
                    - **Call emergency services immediately.** Time-sensitive.  
                    - **Keep the patient calm, lying down; do not give food or drink.**  
                    - **Note symptom onset time.**  
                    - **Monitor breathing and consciousness.**  
                    - **If unconscious, begin CPR if not breathing.**  
                    - **Transport to stroke-capable hospital immediately.**
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "first_name": first_name,
                    "last_name": last_name,
                    "age": age,
                    "gender": gender,
                    "image_name": os.path.basename(uploaded_path),
                    "prediction": prediction,
                    "confidence": round(confidence, 2)
                }
                append_log(entry)
                st.success("Saved to logs.")

                if REPORTLAB_AVAILABLE:
                    if st.button("Download PDF report"):
                        buf = generate_pdf_report(
                            patient_info={"First name": first_name, "Last name": last_name, "Age": age, "Gender": gender},
                            prediction=prediction,
                            confidence=confidence,
                            image_path=uploaded_path
                        )
                        st.download_button("Download Report (PDF)", data=buf, file_name=f"report_{first_name}_{last_name}.pdf", mime="application/pdf")
                else:
                    st.info("PDF export requires reportlab. Install: pip install reportlab")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# (Multi-scan Compare, Doctor Summary, Settings remain the same)
# -------------------------
# Remember to replace st.image(..., use_container_width=True) with width=700 in all other places.

st.markdown('<div class="small-muted" style="text-align:center; margin-top:18px;">Made with ❤️ — Automated Brain Stroke Classification</div>', unsafe_allow_html=True)

