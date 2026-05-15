#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HASYv2 – Reconocedor de Símbolos Matemáticos
Stack: Streamlit + TensorFlow/Keras
Responsive: desktop (2 columnas) + móvil (1 columna)
"""

import os, json
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

IMG_SIZE       = 32
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "modelo_hasy.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_map.json")
TOP_K          = 5

st.set_page_config(
    page_title="HASYv2 · Símbolos Matemáticos",
    page_icon="∑",
    layout="wide",
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #080C14;
  --bg2:        #0D1421;
  --surface:    #111827;
  --surface2:   #1A2332;
  --border:     #1E2D42;
  --border2:    #243347;
  --fg:         #F0F4FF;
  --fg2:        #94A3B8;
  --fg3:        #475569;
  --fg4:        #334155;
  --accent:     #6366F1;
  --accent2:    #818CF8;
  --accent-bg:  rgba(99,102,241,0.10);
  --green:      #10B981;
  --green-bg:   rgba(16,185,129,0.12);
  --amber:      #F59E0B;
  --amber-bg:   rgba(245,158,11,0.12);
  --radius:     16px;
  --radius-sm:  10px;
  --radius-xs:  7px;
}

html, body, .stApp {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--fg);
}

/* ── Ocultar chrome de Streamlit ── */
header[data-testid="stHeader"],
#MainMenu, footer,
[data-testid="stToolbar"] { display: none !important; }

/* Quitar el padding-top de 96px que Streamlit añade al block container */
[data-testid="stMainBlockContainer"] {
  padding: 0 !important;
  max-width: 100% !important;
}
/* Fallback por si el testid cambia */
section.main > div.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── TOPBAR ── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.5rem;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
}
.topbar-brand { display: flex; align-items: center; gap: 10px; }
.brand-icon {
  width: 32px; height: 32px;
  background: var(--accent);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600; font-size: 14px; color: #fff;
  flex-shrink: 0;
}
.brand-name { font-size: 13px; font-weight: 600; color: var(--fg); letter-spacing: -0.01em; }
.brand-sub  { font-size: 10px; color: var(--fg3); font-weight: 400; }
.topbar-pills { display: flex; gap: 6px; }
.pill {
  padding: 3px 10px; border-radius: 9999px;
  font-size: 10px; font-weight: 600; border: 0.5px solid;
  letter-spacing: 0.02em;
}
.pill-accent { background: var(--accent-bg); color: var(--accent2); border-color: rgba(99,102,241,0.3); }
.pill-mono   { background: var(--surface); color: var(--fg3); border-color: var(--border2);
               font-family: 'JetBrains Mono', monospace; }

/* ── PANEL HEADER: st.columns row estilizado como header via :has() ── */
div[data-testid="stHorizontalBlock"]:has(#hdr-title-marker) {
  background: var(--bg2) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0.5rem 1rem !important;
  margin: 0 !important;
  gap: 8px !important;
  align-items: center !important;
}
div[data-testid="stHorizontalBlock"]:has(#hdr-title-marker)
  > div[data-testid="stColumn"] {
  padding: 0 !important;
  min-width: 0 !important;
}
.panel-hdr-title {
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--fg3);
  margin: 0 !important; padding: 0 !important;
  line-height: 30px;
  white-space: nowrap;
}

/* ── CANVAS AREA ── */
.canvas-wrap {
  display: flex; align-items: center; justify-content: center;
  padding: 1.25rem;
}

/* Canvas: usar la clase Streamlit exacta para recortar zona blanca */
[class*="st-key-canvas"] {
  width: 280px !important;
  height: 284px !important;
  overflow: hidden !important;
  border-radius: var(--radius-sm) !important;
  box-shadow: none !important;
  background-color: #000 !important;
  margin: 1rem auto !important;
  display: block !important;
}
[class*="st-key-canvas"] > div {
  width: 280px !important;
  height: 284px !important;
  overflow: hidden !important;
}
[class*="st-key-canvas"] iframe {
  display: block !important;
  width: 280px !important;
  height: 284px !important;
  border: none !important;
}
/* Centrar el canvas en el flujo de la columna */
div[data-testid="element-container"]:has([class*="st-key-canvas"]) {
  display: flex !important;
  justify-content: center !important;
}

/* ── CANVAS FOOTER ── */
.canvas-footer {
  padding: 0.75rem 1rem 1rem;
  border-top: 1px solid var(--border);
}
.canvas-hint {
  font-size: 11px; color: var(--fg4);
  text-align: center; line-height: 1.6; margin-bottom: 10px;
}
.canvas-hint strong { color: var(--fg3); font-weight: 500; }

.stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
.stat-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius-xs); padding: 8px 6px; text-align: center;
}
.stat-val { font-size: 13px; font-weight: 600; color: var(--accent2); font-family: 'JetBrains Mono', monospace; }
.stat-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--fg4); margin-top: 2px; }

/* ── RESULTADO HERO ── */
.section-title {
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--fg3); margin-bottom: 10px;
}

.result-hero {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1rem;
  display: flex; align-items: center; gap: 12px;
}
.sym-box {
  width: 68px; height: 68px; flex-shrink: 0;
  background: var(--accent-bg);
  border: 1px solid rgba(99,102,241,0.2);
  border-radius: 9px;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 4px;
  overflow: hidden;
}
.sym-glyph { font-size: 26px; color: var(--fg); }
.sym-latex { font-size: 9px; color: var(--fg4); font-family: 'JetBrains Mono', monospace;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 64px; }

.result-info { flex: 1; min-width: 0; }
.result-name {
  font-size: 22px; font-weight: 600; color: var(--fg);
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: -0.02em; margin-bottom: 6px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.conf-row { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
.conf-badge {
  font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 9999px;
}
.conf-high { background: var(--green-bg); color: var(--green); border: 0.5px solid rgba(16,185,129,0.3); }
.conf-mid  { background: var(--accent-bg); color: var(--accent2); border: 0.5px solid rgba(99,102,241,0.3); }
.conf-low  { background: var(--amber-bg); color: var(--amber); border: 0.5px solid rgba(245,158,11,0.3); }
.conf-pct  { font-size: 13px; font-weight: 600; color: var(--fg2); font-family: 'JetBrains Mono', monospace; }
.bar-track { height: 4px; background: var(--border2); border-radius: 9999px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 9999px; transition: width 0.4s cubic-bezier(0.4,0,0.2,1); }

/* ── TOP-K ── */
.topk-list { display: flex; flex-direction: column; }
.topk-row {
  display: grid;
  grid-template-columns: 18px minmax(0, 3.5rem) 1fr 3rem;
  align-items: center; gap: 8px;
  padding: 7px 0;
  border-bottom: 0.5px solid var(--border);
}
.topk-row:last-child { border-bottom: none; }
.tk-rank { font-size: 10px; color: var(--fg4); font-family: 'JetBrains Mono', monospace; text-align: right; }
.tk-sym  { font-size: 13px; color: var(--fg2); font-family: 'JetBrains Mono', monospace;
           overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tk-sym.first { color: var(--accent2); }
.tk-bar-bg { height: 4px; background: var(--border2); border-radius: 9999px; overflow: hidden; }
.tk-bar    { height: 100%; border-radius: 9999px; }
.tk-pct    { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: var(--fg3); text-align: right; }

/* ── EMPTY STATE ── */
.empty {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  min-height: 260px; gap: 10px;
  color: var(--fg4); text-align: center;
}
.empty-icon {
  width: 52px; height: 52px;
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: var(--radius-xs);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: var(--fg3);
}
.empty p    { font-size: 13px; color: var(--fg2); }
.empty span { font-size: 11px; color: var(--fg4); }

/* ── BOTONES UNDO/TRASH ── */
div[data-testid="stHorizontalBlock"]:has(#hdr-title-marker) button {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--fg3) !important;
  border-radius: 7px !important;
  width: 30px !important; height: 30px !important;
  min-width: 30px !important;
  padding: 0 !important;
  font-size: 1.1rem !important;
  display: flex !important;
  align-items: center !important; justify-content: center !important;
  transition: all 0.15s !important;
}
div[data-testid="stHorizontalBlock"]:has(#hdr-title-marker) button:hover {
  border-color: var(--accent) !important;
  background: var(--accent-bg) !important;
  color: var(--accent2) !important;
}
div[data-testid="stHorizontalBlock"]:has(#hdr-title-marker) button:active {
  transform: scale(0.93) !important;
}

/* ── Streamlit columns gap reset ── */
div[data-testid="stHorizontalBlock"] {
  gap: 6px !important;
  align-items: center !important;
}
div[data-testid="column"] {
  padding: 0 !important;
}

/* ── Columnas externas: fondo y separador ── */
section.main > div.block-container
  > div > div > div[data-testid="stHorizontalBlock"]
  > div[data-testid="stColumn"]:first-child {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  padding: 0 !important;
}
section.main > div.block-container
  > div > div > div[data-testid="stHorizontalBlock"]
  > div[data-testid="stColumn"]:last-child {
  background: var(--bg) !important;
  padding: 1.25rem 1.5rem !important;
}

/* ── RESPONSIVE MÓVIL ── */
@media (max-width: 768px) {
  /* Colapsar SOLO el stHorizontalBlock de las columnas principales */
  [data-testid="stHorizontalBlock"].st-emotion-cache-r3ry0f,
  [data-testid="stHorizontalBlock"]:not(:has([id="hdr-title-marker"])) {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
  }
  [data-testid="stHorizontalBlock"].st-emotion-cache-r3ry0f
    > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"]:not(:has([id="hdr-title-marker"]))
    > [data-testid="stColumn"] {
    width: 100% !important;
    flex: 0 0 100% !important;
    max-width: 100% !important;
    min-width: 100% !important;
    border-right: none !important;
  }
  /* Topbar compacto */
  .topbar-pills .pill-mono { display: none; }
  .topbar { padding: 0.6rem 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ── KaTeX ─────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js" defer></script>
""", unsafe_allow_html=True)

# ── Resources ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_map():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v["latex"] for k, v in raw.items()}

# ── State ─────────────────────────────────────────────────────────────────────
for key, val in [
    ("canvas_key", 0),
    ("canvas_history", []),
    ("last_json", None),
    ("current_initial_drawing", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

model     = load_model()
class_map = load_class_map()

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess_canvas(img: np.ndarray):
    gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
    if np.max(gray) < 10:
        return None
    gray   = 255 - gray
    coords = cv2.findNonZero(255 - gray)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    pad = 20
    x = max(x - pad, 0); y = max(y - pad, 0)
    w = min(w + 2 * pad, gray.shape[1] - x)
    h = min(h + 2 * pad, gray.shape[0] - y)
    crop = gray[y:y+h, x:x+w]
    side = max(crop.shape)
    sq   = np.full((side, side), 255, dtype=np.uint8)
    dy, dx = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    sq[dy:dy+crop.shape[0], dx:dx+crop.shape[1]] = crop
    rsz = cv2.resize(sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (rsz.astype("float32") / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def latex_to_display(latex: str) -> str:
    simple = {
        r"\sqrt{}": "√", r"\leftarrow": "←", r"\rightarrow": "→",
        r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
        r"\epsilon": "ε", r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ",
        r"\pi": "π", r"\sigma": "σ", r"\phi": "φ", r"\omega": "ω",
        r"\Gamma": "Γ", r"\Delta": "Δ", r"\Sigma": "Σ", r"\Omega": "Ω",
        r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
        r"\checkmark": "✓", r"\times": "×", r"\div": "÷",
        r"\pm": "±", r"\leq": "≤", r"\geq": "≥", r"\neq": "≠",
        r"\approx": "≈", r"\in": "∈", r"\subset": "⊂", r"\cup": "∪",
        r"\cap": "∩", r"\forall": "∀", r"\exists": "∃",
        r"\int": "∫", r"\sum": "∑", r"\prod": "∏",
    }
    return simple.get(latex, latex)

# ── TOPBAR (único st.markdown autocontenido) ───────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="brand-icon">∑</div>
    <div>
      <div class="brand-name">HASYv2</div>
      <div class="brand-sub">Reconocedor de Símbolos Matemáticos</div>
    </div>
  </div>
  <div class="topbar-pills">
    <span class="pill pill-accent">369 clases</span>
    <span class="pill pill-mono">CNN · 32×32</span>
    <span class="pill pill-mono">TensorFlow</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT: st.columns es el único grid padre ─────────────────────────────────
col_l, col_r = st.columns([4, 5], gap="large")

# ═══════════════════════════════════════════════════════
# PANEL IZQUIERDO – Canvas
# ═══════════════════════════════════════════════════════
with col_l:

    # Header: título + botones en un ÚNICO st.columns — mismo nivel horizontal
    hdr_title_col, hdr_undo_col, hdr_clear_col = st.columns([6, 1, 1])
    with hdr_title_col:
        # El ID es el ancla CSS para :has(#hdr-title-marker)
        st.markdown('<p class="panel-hdr-title" id="hdr-title-marker">Área de dibujo</p>',
                    unsafe_allow_html=True)
    with hdr_undo_col:
        if st.button("↺", help="Deshacer último trazo", key="undo_btn"):
            if st.session_state.canvas_history:
                st.session_state.canvas_history.pop()
                st.session_state.current_initial_drawing = (
                    st.session_state.canvas_history[-1]
                    if st.session_state.canvas_history else None
                )
                st.session_state.last_json = st.session_state.current_initial_drawing
                st.session_state.canvas_key += 1
                st.rerun()
    with hdr_clear_col:
        if st.button("🗑", help="Limpiar lienzo", key="clear_btn"):
            st.session_state.canvas_history          = []
            st.session_state.current_initial_drawing = None
            st.session_state.last_json               = None
            st.session_state.canvas_key             += 1
            st.rerun()

    # Canvas (sin divs envolventes)
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=14,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        initial_drawing=st.session_state.current_initial_drawing,
        display_toolbar=False,
    )

    # Historial de trazos
    if canvas_result.json_data is not None:
        cj = canvas_result.json_data
        if cj != st.session_state.last_json:
            if cj.get("objects"):
                st.session_state.canvas_history.append(cj)
            st.session_state.last_json = cj

    # Footer autocontenido
    st.markdown("""
<div class="canvas-footer">
  <p class="canvas-hint">
    <strong>Dibuja</strong> un símbolo matemático con el ratón o dedo.
    El modelo lo clasificará en tiempo real.
  </p>
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-val">369</div>
      <div class="stat-lbl">Clases</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">32px</div>
      <div class="stat-lbl">Resolución</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">Top‑5</div>
      <div class="stat-lbl">Predicciones</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PANEL DERECHO – Resultados
# ═══════════════════════════════════════════════════════
with col_r:

    has_result = False

    if canvas_result.image_data is not None:
        processed = preprocess_canvas(canvas_result.image_data)
        if processed is not None:
            has_result = True
            preds      = model.predict(processed, verbose=0)
            top_idxs   = np.argsort(preds[0])[::-1][:TOP_K]
            best_idx   = top_idxs[0]
            best_label = class_map.get(best_idx, "?")
            best_conf  = float(preds[0][best_idx])
            pct        = int(best_conf * 100)
            glyph      = latex_to_display(best_label)

            if best_conf >= 0.80:
                badge_cls = "conf-high"; badge_txt = "Alta confianza"; bar_color = "#10B981"
            elif best_conf >= 0.50:
                badge_cls = "conf-mid";  badge_txt = "Confianza media"; bar_color = "#6366F1"
            else:
                badge_cls = "conf-low";  badge_txt = "Confianza baja";  bar_color = "#F59E0B"

            # ── Hero (único st.markdown autocontenido) ──
            result_html = (
                '<div>'
                '<div class="section-title">Resultado del modelo</div>'
                '<div class="result-hero">'
                '<div class="sym-box">'
                '<div class="sym-glyph">' + glyph + '</div>'
                '<div class="sym-latex">' + best_label + '</div>'
                '</div>'
                '<div class="result-info">'
                '<div class="result-name">' + glyph + '&nbsp;'
                '<span style="font-size:13px;color:var(--fg3);">' + best_label + '</span>'
                '</div>'
                '<div class="conf-row">'
                '<span class="conf-badge ' + badge_cls + '">' + badge_txt + '</span>'
                '<span class="conf-pct">' + f"{best_conf:.1%}" + '</span>'
                '</div>'
                '<div class="bar-track">'
                '<div class="bar-fill" style="width:' + str(pct) + '%;background:' + bar_color + ';"></div>'
                '</div>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(result_html, unsafe_allow_html=True)

            # ── Top-K (construida como lista y unida, sin f-strings anidados) ──
            rows = []
            for rank, idx in enumerate(top_idxs, 1):
                lbl  = class_map.get(idx, "?")
                conf = float(preds[0][idx])
                rp   = int(conf * 100)
                g    = latex_to_display(lbl)
                bar  = "#6366F1" if rank == 1 else "#243347"
                cls  = " first" if rank == 1 else ""
                rows.append(
                    '<div class="topk-row">'
                    '<span class="tk-rank">' + str(rank) + '</span>'
                    '<span class="tk-sym' + cls + '">' + g + '</span>'
                    '<div class="tk-bar-bg">'
                    '<div class="tk-bar" style="width:' + str(rp) + '%;background:' + bar + ';"></div>'
                    '</div>'
                    '<span class="tk-pct">' + f"{conf:.0%}" + '</span>'
                    '</div>'
                )
            topk_html = (
                '<div>'
                '<div class="section-title">Top ' + str(TOP_K) + ' candidatos</div>'
                '<div class="topk-list">'
                + "".join(rows) +
                '</div>'
                '</div>'
            )
            st.markdown(topk_html, unsafe_allow_html=True)

    if not has_result:
        st.markdown("""
<div class="empty">
  <div class="empty-icon">∫</div>
  <p>Dibuja un símbolo para comenzar</p>
  <span>El modelo CNN analizará tu trazo al instante</span>
</div>
""", unsafe_allow_html=True)
