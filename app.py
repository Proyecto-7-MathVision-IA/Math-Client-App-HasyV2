#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
  HASYv2 – Aplicación Web de Reconocimiento de Símbolos Matemáticos
====================================================================
  Modelo:   modelo_hasy.keras (CNN, 369 clases, 32×32 px)
  Stack:    Streamlit + TensorFlow/Keras
  Ejecutar: streamlit run app.py
====================================================================
"""

import os
import json

# pyrefly: ignore [missing-import]
import streamlit as st
import tensorflow as tf
# pyrefly: ignore [missing-import]
from streamlit_drawable_canvas import st_canvas
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

# ──────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────
IMG_SIZE = 32
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_hasy.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_map.json")
TOP_K = 5  # Número de predicciones candidatas a mostrar

# ──────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HASYv2 – Reconocedor de Símbolos",
    page_icon="✏️",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (cacheados para no recargar en cada interacción)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Carga el modelo .keras entrenado."""
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_map():
    """
    Carga el mapeo index → { symbol_id, latex }.
    Devuelve un dict  int → str(latex).
    """
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v["latex"] for k, v in raw.items()}


model = load_model()
class_map = load_class_map()


# ──────────────────────────────────────────────────────────────────
# PREPROCESADO DE IMAGEN
# ──────────────────────────────────────────────────────────────────
def preprocess_canvas(image_data: np.ndarray) -> np.ndarray | None:
    """
    Convierte la imagen RGBA del canvas (280×280) al formato
    que espera el modelo: escala de grises, 32×32, normalizada [0,1].

    El canvas tiene fondo negro (0) y trazo blanco (255).
    HASYv2 fue entrenado con fondo blanco (255) y trazo negro (0),
    así que invertimos los colores.

    Devuelve None si el canvas está vacío (nada dibujado).
    """
    # Convertir RGBA → escala de grises
    gray = cv2.cvtColor(image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Comprobar si hay algo dibujado
    if np.max(gray) < 10:
        return None

    # Invertir: el modelo espera trazo negro sobre fondo blanco
    gray = 255 - gray

    # Recortar al bounding box del trazo + padding
    coords = cv2.findNonZero(255 - gray)  # píxeles oscuros (trazo)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    padding = 20
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, gray.shape[1] - x)
    h = min(h + 2 * padding, gray.shape[0] - y)
    cropped = gray[y : y + h, x : x + w]

    # Hacer la imagen cuadrada (rellenar con blanco)
    side = max(cropped.shape)
    square = np.full((side, side), 255, dtype=np.uint8)
    dy = (side - cropped.shape[0]) // 2
    dx = (side - cropped.shape[1]) // 2
    square[dy : dy + cropped.shape[0], dx : dx + cropped.shape[1]] = cropped

    # Redimensionar a 32×32
    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Normalizar a [0, 1] y dar forma (1, 32, 32, 1) para el modelo
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ──────────────────────────────────────────────────────────────────
# INTERFAZ
# ──────────────────────────────────────────────────────────────────
st.title("✏️ Reconocedor de Símbolos Matemáticos")
st.markdown(
    "Dibuja un símbolo matemático (letra, número, operador…) en el recuadro "
    "y el modelo **HASYv2** lo identificará entre **369 clases** posibles."
)

# ── Canvas de dibujo ──
col_canvas, col_result = st.columns([1, 1])

with col_canvas:
    st.subheader("🖊️ Dibuja aquí")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=12,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# ── Predicción ──
with col_result:
    st.subheader("🔍 Predicción")

    if canvas_result.image_data is not None:
        processed = preprocess_canvas(canvas_result.image_data)

        if processed is not None:
            # Inferencia
            predictions = model.predict(processed, verbose=0)
            top_indices = np.argsort(predictions[0])[::-1][:TOP_K]

            # Resultado principal
            best_idx = top_indices[0]
            best_label = class_map.get(best_idx, "?")
            best_conf = predictions[0][best_idx]

            st.metric(label="Símbolo detectado", value=best_label)

            if best_conf < 0.50:
                st.warning(f"Confianza baja: **{best_conf:.1%}**. Intenta dibujar más claro.")
            elif best_conf < 0.80:
                st.info(f"Confianza media: **{best_conf:.1%}**")
            else:
                st.success(f"Confianza alta: **{best_conf:.1%}**")

            # Top-K candidatos
            st.markdown("---")
            st.markdown(f"**Top {TOP_K} candidatos:**")
            for rank, idx in enumerate(top_indices, start=1):
                label = class_map.get(idx, "?")
                conf = predictions[0][idx]
                bar = "█" * int(conf * 20)
                st.text(f"  {rank}. {label:<12s}  {conf:6.2%}  {bar}")

            # Gráfico de barras de las top predicciones
            st.markdown("---")
            chart_data = {
                class_map.get(i, "?"): float(predictions[0][i])
                for i in top_indices
            }
            st.bar_chart(chart_data)
        else:
            st.info("Dibuja un símbolo en el recuadro de la izquierda para comenzar.")
    else:
        st.info("Dibuja un símbolo en el recuadro de la izquierda para comenzar.")

# ── Footer ──
st.markdown("---")
st.caption(
    "Modelo CNN entrenado con el dataset **HASYv2** (168.236 imágenes, 369 clases). "
    "Arquitectura: 3 bloques Conv + BatchNorm + Dropout → Dense 256 → Softmax."
)