# app_final_deploy_full.py
# Automated Brain Stroke Classification — Deployment-ready Full App
# Features: MONAI DenseNet121 prediction, multi-scan comparison, doctor summary,
# PDF export, temp uploads, runtime model + example CT download, updated messages.

import os
import io
import time
import tempfile
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

# PDF library
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

import gdown

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Automated Brain Stroke Classification", page_icon="🧠", layout="wide")

MODEL_URL = "https://drive.google.com/file/d/1BXWGOpZ4_9WSx_C5pZwX2RduHCGe4-24/view?usp=sharing"  # Replace with your model
MODEL_PATH = "best_monai_densenet121.pth"

EXAMPLE_IMAGE_URL = "https://drive.google.com/file/d/1Ba7xkNPGIskgLeMUkpJ7y0385NH_ps3R/view?usp=sharing"  # Replace with example CT
EXAMPLE_IMAGE_PATH = "100 (20)"

LOGS_CSV = "predictions_log.csv"

single_transform = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])
classes = ["Normal", "Stroke"]

# -------------------------
# Download model & example CT
# -------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

if not os.path.exists(EXAMPLE_IMAGE_PATH):
    st.info("Downloading example CT image...")
    gdown.download(EXAMPLE_IMAGE_URL, EXAMPLE_IMAGE_PATH, quiet=False)
    st.success("Example CT downloaded!")

# -------------------------
# Styles
# -------------------------
st.markdown(
    """
    <style>
    body { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); }
    .app-title { font-size: 42px; font-weight:900; text-align:center;
                 background: linear-gradient(90deg,#ff6b6b,#f6d365,#5ee7df);
                 -webkit-background-clip: text; color: transparent; margin-bottom: -6px; }
    .card { background: white; padding:18px; border-radius:12px; box-shadow: 0 6px 20px rgba(44,62,80,0.06); }
    .prec { background: linear-gradient(90deg,#fff1f0,#fff7ed); padding:14px; border-radius:10px; }
    .small-muted { color:#64748b; font-size:13px; }
    .loader { border: 6px solid #f3f3f3; border-top: 6px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin:auto; }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="app-title">🧠 Automated Brain Stroke Classification</div>', unsafe_allow_html=True)

# -------------------------
# Sidebar navigation
# -------------------------
page = st.sidebar.selectbox("Navigation", ["Classification", "Multi-scan Compare", "Doctor Summary", "Settings"])

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
    device = "cpu"
    load_error_text = str(e)

# -------------------------
# Utils
# -------------------------
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def predict_with_monai(path):
    img = single_transform(path)
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], float(conf.item()*100), probs.cpu().numpy()[0]

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

def generate_pdf_report(patient_info: dict, prediction: str, confidence: float, image_path: str, notes=None):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not installed.")
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(595,842))
    width, height = 595,842
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height-60, "Automated Brain Stroke Classification")
    c.setFont("Helvetica", 11)
    c.drawString(40, height-85, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, height-120, "Patient Information")
    c.setFont("Helvetica", 11)
    y = height-140
    for k,v in patient_info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 16
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y-6, "Prediction")
    c.setFont("Helvetica", 12)
    c.drawString(50, y-26, f"Result: {prediction}")
    c.drawString(50, y-44, f"Confidence: {confidence:.2f}%")
    if notes:
        y -= 60
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Doctor Notes:")
        y -= 16
        c.setFont("Helvetica", 11)
        for line in str(notes).splitlines():
            c.drawString(50, y, line)
            y -= 14
    try:
        thumb = Image.open(image_path).convert("RGB")
        thumb.thumbnail((240,240))
        thumb_buf = io.BytesIO()
        thumb.save(thumb_buf, format="PNG")
        thumb_buf.seek(0)
        c.drawInlineImage(thumb_buf, width-300, height-350, 240,240)
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
        st.error("Model not loaded. Check Settings.")
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
    c1, c2 = st.columns([3,1])
    uploaded = c1.file_uploader("Upload CT scan image (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=False)
    if c2.button("Use example CT"):
        uploaded_path = EXAMPLE_IMAGE_PATH
        st.success("Loaded example CT.")
    else:
        uploaded_path = None

    if uploaded is not None:
        uploaded_path = save_uploaded_file(uploaded)

    if uploaded_path:
        st.image(uploaded_path, caption="CT preview", use_container_width=True)
        if st.button("Run Prediction"):
            with st.spinner("Running model..."):
                try:
                    prediction, confidence, probs = predict_with_monai(uploaded_path)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    prediction, confidence, probs = None, None, None

            if prediction is not None:
                st.markdown("### Result")
                if prediction == "Normal":
                    st.success(f"🟢 Prediction: {prediction} — Confidence: {confidence:.2f}%")
                    st.info("✅ Your CT scan appears normal. Continue routine health checkups. If you feel unwell, consult a doctor.")
                else:
                    st.error(f"🔴 Prediction: {prediction} — Confidence: {confidence:.2f}%")
                    st.markdown('<div class="prec">', unsafe_allow_html=True)
                    st.markdown("### Immediate precautions & next steps")
                    st.markdown("""
                    - **Call emergency services immediately.** Time is critical.  
                    - **Keep the patient calm, lying down; do not give food or drink.**  
                    - **Note the symptom onset time.**  
                    - **Monitor breathing and consciousness continuously.**  
                    - **If unconscious, begin CPR if patient is not breathing.**  
                    - **Transport to a stroke-capable hospital immediately.**
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Save log
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

                # PDF export
                if REPORTLAB_AVAILABLE and st.button("Download PDF report"):
                    buf = generate_pdf_report(
                        patient_info={"First name": first_name, "Last name": last_name, "Age": age, "Gender": gender},
                        prediction=prediction,
                        confidence=confidence,
                        image_path=uploaded_path
                    )
                    st.download_button("Download Report (PDF)", data=buf, file_name=f"report_{first_name}_{last_name}.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PAGE: Multi-scan Compare
# -------------------------
elif page == "Multi-scan Compare":
    st.header("Multi-scan Comparison")
    st.markdown("Upload multiple CT scans (same patient or multiple). Batch predictions with comparison chart.")
    multi = st.file_uploader("Upload multiple CTs", type=["png","jpg","jpeg"], accept_multiple_files=True)

    if st.button("Run Batch Prediction") and (multi is not None and len(multi) > 0):
        results = []
        for up in multi:
            tmpname = save_uploaded_file(up)
            try:
                pred, conf, probs = predict_with_monai(tmpname)
            except Exception:
                pred, conf = "ERR", 0.0
            results.append({"image": up.name, "prediction": pred, "confidence": round(conf,2)})
        df = pd.DataFrame(results)
        st.dataframe(df)
        fig, ax = plt.subplots(figsize=(8,3))
        df_plot = df.copy()
        df_plot["confidence"] = df_plot["confidence"].astype(float)
        ax.bar(df_plot["image"], df_plot["confidence"], color=['#2ecc71' if p=="Normal" else '#e74c3c' for p in df_plot["prediction"]])
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0,100)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.info("Upload multiple CT images and press 'Run Batch Prediction'.")

# -------------------------
# PAGE: Doctor Summary
# -------------------------
elif page == "Doctor Summary":
    st.header("Doctor Summary")
    st.markdown("View recent predictions, add clinical notes, and export a PDF summary.")

    df_logs = load_logs_df()
    if df_logs.shape[0]==0:
        st.info("No predictions logged yet.")
    else:
        st.dataframe(df_logs.tail(50))
        sel_idx = st.number_input("Select row number (0 = first row) to review", min_value=0, max_value=max(0,len(df_logs)-1), value=len(df_logs)-1)
        if st.button("Load selected"):
            row = df_logs.iloc[sel_idx].to_dict()
            st.write("Selected record:")
            st.json(row)
            notes = st.text_area("Doctor notes / recommendations")
            if st.button("Export doctor summary (PDF)"):
                if REPORTLAB_AVAILABLE:
                    buffer = generate_pdf_report(patient_info=row, prediction=row["prediction"],
                                                 confidence=row["confidence"], image_path=row["image_name"],
                                                 notes=notes)
                    st.download_button("Download Doctor Summary PDF", data=buffer, file_name="doctor_summary.pdf", mime="application/pdf")
                else:
                    st.error("reportlab not installed. Install: pip install reportlab")

# -------------------------
# PAGE: Settings
# -------------------------
elif page == "Settings":
    st.header("Settings")
    st.write("Model path:", MODEL_PATH)
    if model is None:
        st.error("Model not loaded.")
        if 'load_error_text' in locals():
            st.write("Error:", load_error_text)
    else:
        st.success("Model loaded OK.")
    st.markdown("---")
    st.write("Logs:")
    st.write(f"Saved to `{LOGS_CSV}` in app folder.")
    if os.path.exists(LOGS_CSV):
        st.download_button("Download logs CSV", data=open(LOGS_CSV,"rb"), file_name="predictions_log.csv")

# -------------------------
st.markdown('<div class="small-muted" style="text-align:center; margin-top:18px;">Made with ❤️ — Automated Brain Stroke Classification</div>', unsafe_allow_html=True)

