"""SteelScan: demo klasifikasi cacat permukaan baja (NEU) dengan Grad-CAM dan log SQLite."""
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import steel_utils as su

DB_PATH = "predictions.db"
MODEL_PATH = "steel_cnn.keras"
LOW_CONF = 0.70       # di bawah ini, sarankan pemeriksaan manual
ALERT_WINDOW = 20     # jumlah inspeksi terakhir yang dipantau
ALERT_SHARE = 0.40    # ambang proporsi satu jenis cacat

st.set_page_config(page_title="SteelScan", page_icon="🔩", layout="wide")


@st.cache_resource
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS predictions (ts TEXT, filename TEXT, label TEXT, confidence REAL)")
    return con


def save_prediction(filename, label, conf):
    con = _connect()
    try:
        con.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), filename, label, float(conf)),
        )
        con.commit()
    finally:
        con.close()


def read_history():
    con = _connect()
    try:
        return pd.read_sql("SELECT * FROM predictions ORDER BY ts DESC", con)
    finally:
        con.close()


st.title("🔩 SteelScan")
st.caption("Klasifikasi 6 jenis cacat permukaan baja canai panas (dataset NEU). Demo edukasi, bukan sistem inspeksi produksi.")
tab_scan, tab_hist, tab_about = st.tabs(["Inspeksi", "Riwayat dan peringatan", "Tentang"])

with tab_scan:
    up = st.file_uploader("Unggah gambar permukaan baja (grayscale, mirip data NEU)", type=["bmp", "jpg", "jpeg", "png"])
    if up is not None:
        raw = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            st.error("File tidak bisa dibaca sebagai gambar.")
        else:
            model = load_model()
            pre = su.preprocess_gray(raw)
            rgb = su.to_rgb(pre)
            proba = model.predict(rgb[None], verbose=0)[0]
            cam, cls, conf = su.grad_cam(model, rgb)
            label = su.CLASSES[cls]

            c1, c2, c3 = st.columns(3)
            c1.image(pre, caption="Setelah preprocessing (CLAHE + resize)", clamp=True)
            c2.image(su.overlay_cam(pre, cam), caption="Grad-CAM: area yang memengaruhi prediksi")
            c3.metric("Prediksi", label)
            c3.write(f"Confidence: {conf:.1%}")
            if conf < LOW_CONF:
                c3.warning("Confidence rendah, periksa manual.")
            st.bar_chart(pd.Series(proba, index=su.CLASSES, name="probabilitas"))

            if st.button("Simpan ke riwayat"):
                save_prediction(up.name, label, conf)
                st.success("Tersimpan.")

with tab_hist:
    hist = read_history()
    if hist.empty:
        st.info("Belum ada riwayat. Simpan hasil dari tab Inspeksi.")
    else:
        st.bar_chart(hist["label"].value_counts())
        recent = hist.head(ALERT_WINDOW)
        share = recent["label"].value_counts(normalize=True)
        if len(recent) >= 10 and share.iloc[0] >= ALERT_SHARE:
            st.warning(
                f"'{share.index[0]}' mencapai {share.iloc[0]:.0%} dari {len(recent)} inspeksi terakhir. "
                "Periksa proses produksi."
            )
        st.dataframe(hist.head(50), use_container_width=True)

with tab_about:
    st.markdown(
        """
**Pendekatan.** MobileNetV2 (transfer learning, dua tahap) pada gambar yang diproses dengan CLAHE (OpenCV).
Dibandingkan dengan baseline HOG + SVM, divalidasi silang, dan diuji ketahanannya terhadap perubahan kecerahan, noise, dan blur.

**Keterbatasan.** Data kecil (1.800 gambar), kondisi terkurasi, belum diuji di lini produksi.
Grad-CAM adalah alat bantu interpretasi, bukan bukti kausal. Gambar di luar domain NEU akan menghasilkan prediksi yang tidak bermakna.

**Data.** NEU surface defect database (versi Kaggle). Lihat README untuk sumber dan lisensi.
"""
    )
