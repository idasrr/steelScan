"""Fungsi bersama untuk notebook training dan aplikasi Streamlit (SteelScan)."""
import os
import re

import cv2
import numpy as np

IMG_SIZE = 224
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Nama folder atau awalan nama file yang dikenali (kode NEU: Cr, In, Pa, PS, RS, Sc)
_ALIASES = {
    "crazing": ["crazing", "cr", "cracks"],
    "inclusion": ["inclusion", "in"],
    "patches": ["patches", "pa"],
    "pitted_surface": ["pitted_surface", "pitted", "ps"],
    "rolled-in_scale": ["rolled-in_scale", "rolled", "rs"],
    "scratches": ["scratches", "sc"],
}


def _norm(s):
    return re.sub(r"[\-\s]+", "_", s.lower())


_LOOKUP = {_norm(a): c for c, names in _ALIASES.items() for a in names}


def label_from_path(path):
    """Kelas dari nama folder induk; kalau tidak cocok, dari awalan nama file."""
    folder = _LOOKUP.get(_norm(os.path.basename(os.path.dirname(str(path)))))
    if folder:
        return folder
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    base = re.sub(r"[\d_\-\s]+$", "", stem)
    return _LOOKUP.get(_norm(base))


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    return img


def preprocess_gray(img_gray):
    """CLAHE untuk menormalkan kontras lokal, lalu resize ke IMG_SIZE x IMG_SIZE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img_gray)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def load_image(path):
    return preprocess_gray(read_gray(path))


def to_rgb(x):
    """Grayscale uint8 (H,W) atau (N,H,W) menjadi 3 kanal (...,H,W,3) uint8."""
    x = np.asarray(x, dtype=np.uint8)
    return np.repeat(x[..., None], 3, axis=-1)


def grad_cam(model, rgb):
    """Grad-CAM dari konvolusi terakhir backbone. rgb: (H,W,3) nilai 0..255.
    Mengembalikan (cam 0..1 ukuran IMG_SIZE, indeks kelas prediksi, confidence)."""
    import tensorflow as tf

    x = tf.convert_to_tensor(rgb[None], dtype=tf.float32) / 127.5 - 1.0
    backbone = model.get_layer("backbone")
    with tf.GradientTape() as tape:
        fmap = backbone(x, training=False)
        tape.watch(fmap)
        probs = model.get_layer("cls")(model.get_layer("gap")(fmap))
        cls = int(tf.argmax(probs[0]))
        score = probs[:, cls]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam, cls, float(probs[0, cls])


def overlay_cam(gray, cam, alpha=0.4):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=-1)
    return np.uint8((1 - alpha) * base + alpha * heat)
