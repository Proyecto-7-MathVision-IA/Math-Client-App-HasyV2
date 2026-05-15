#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HASYv2 – Reconocedor de Símbolos Matemáticos
Stack: Streamlit + TensorFlow/Keras
"""

import os, json
import streamlit.components.v1 as components
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

IMG_SIZE   = 32
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_hasy.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_map.json")
TOP_K = 5

st.set_page_config(page_title="HASYv2 · Símbolos Matemáticos", page_icon="∑", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
@import url('https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #080C14;
  --bg2:       #0D1421;
  --surface:   #111827;
  --surface2:  #1A2332;
  --border:    #1E2D42;
  --border2:   #243347;
  --fg:        #F0F4FF;
  --fg2:       #94A3B8;
  --fg3:       #4B6080;
  --accent:    #6366F1;
  --accent2:   #818CF8;
  --accent-bg: rgba(99,102,241,0.12);
  --green:     #10B981;
  --green-bg:  rgba(16,185,129,0.12);
  --amber:     #F59E0B;
  --amber-bg:  rgba(245,158,11,0.12);
  --red:       #EF4444;
  --radius:    18px;
  --radius-sm: 12px;
  --radius-xs: 8px;
}

html, body, .stApp {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--fg);
}

/* ── CUSTOM BUTTONS (PRO MAX) ── */
div[data-testid="stHorizontalBlock"] button {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--fg) !important;
  border-radius: 10px !important;
  width: 38px !important;
  height: 38px !important;
  min-width: 38px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0 !important;
  font-size: 1.3rem !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

div[data-testid="stHorizontalBlock"] button:hover {
  border-color: var(--accent) !important;
  background: var(--accent-bg) !important;
  color: var(--accent2) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 16px rgba(99,102,241,0.25) !important;
}

div[data-testid="stHorizontalBlock"] button:active {
  transform: translateY(0) scale(0.92) !important;
}

/* Tooltip style override for help icons */
div[data-testid="stTooltipHoverTarget"] {
  display: flex;
  align-items: center;
}


/* Hide Streamlit chrome */
header[data-testid="stHeader"],
#MainMenu,
footer,
[data-testid="stToolbar"] { display: none !important; }

section.main > div.block-container {
  padding: 1rem 2rem !important;
  max-width: 100% !important;
  padding-top: 0rem !important;
}

/* ── NATIVE COLUMN PANEL (LEFT) ── */
/* We target the immediate vertical block containing our marker */
div[data-testid="stVerticalBlock"]:has(> div > div > div > #left-panel-marker),
div[data-testid="stColumn"]:has(#left-panel-marker) {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
  padding: 0 !important;
}

/* Left Panel Header Row */
div[data-testid="stColumn"]:has(#left-panel-marker) > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:first-of-type {
  background: var(--bg2) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0.8rem 1.4rem !important;
  align-items: center !important;
}

/* Fix Canvas White Background & Centering */
div[data-testid="stVerticalBlock"]:has(#left-panel-marker) iframe,
div[data-testid="stColumn"]:has(#left-panel-marker) iframe {
  display: block !important;
  width: 320px !important;
  max-width: 320px !important;
  margin: 1.5rem auto !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  box-shadow: 0 0 0 1px var(--border2), 0 0 40px rgba(99,102,241,0.08) !important;
  background-color: #000 !important;
}

/* Flexbox for Header Buttons */
div[data-testid="stColumn"]:has(#left-panel-marker) > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="stColumn"]:nth-child(2) [data-testid="stElementContainer"] {
  width: auto !important;
}
  width: auto !important;
}

div[data-testid="stVerticalBlock"]:has(#left-panel-marker) [data-testid="stElementContainer"] {
    display: flex;
    justify-content: center;
    width: 100%;
}

/* ── PAGE SHELL ── */
.page {
  display: grid;
  grid-template-rows: auto 1fr;
  min-height: 100vh;
  background: var(--bg);
}

/* ── TOPBAR ── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 2rem;
  border-bottom: 1px solid var(--border);
  background: rgba(8,12,20,0.85);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 100;
}
.topbar-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.brand-icon {
  width: 38px; height: 38px;
  background: linear-gradient(135deg, #6366F1, #818CF8);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600; font-size: 1.1rem;
  color: #fff;
  box-shadow: 0 0 20px rgba(99,102,241,0.4);
  flex-shrink: 0;
}
.brand-name {
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--fg);
}
.brand-sub {
  font-size: 0.72rem;
  color: var(--fg2);
  font-weight: 400;
}
.topbar-pills {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.pill {
  padding: 0.3rem 0.8rem;
  border-radius: 9999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.pill-accent { background: var(--accent-bg); color: var(--accent2); border: 1px solid rgba(99,102,241,0.25); }
.pill-mono   { background: var(--surface); color: var(--fg2); border: 1px solid var(--border2);
               font-family: 'JetBrains Mono', monospace; }

/* ── PANEL HEADER PRO ── */
.panel-header-pro {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.8rem 1.4rem;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  border-top-left-radius: 18px;
  border-top-right-radius: 18px;
  min-height: 60px;
}
.panel-header-pro .panel-title {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg2);
}

/* ── CONTENT GRID ── */
.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  padding: 1.5rem 2rem 2.5rem;
  align-items: start;
}
@media (max-width: 860px) {
  .content-grid { grid-template-columns: 1fr; padding: 1rem; }
  .topbar-pills { display: none; }
}

/* ── PANEL ── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.1rem 1.4rem;
  border-bottom: 1px solid var(--border);
  background: var(--bg2);
}
.panel-title {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg2);
}
.panel-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
}
.panel-body { padding: 1.5rem; }

/* ── CANVAS ── */
.canvas-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}
.canvas-frame {
  position: relative;
  border-radius: var(--radius-sm);
  overflow: hidden;
  box-shadow: 0 0 0 1px var(--border2), 0 0 40px rgba(99,102,241,0.08);
}
.canvas-frame::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: var(--radius-sm);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
  z-index: 1;
  pointer-events: none;
}
.canvas-hint {
  font-size: 0.78rem;
  color: var(--fg3);
  text-align: center;
  line-height: 1.5;
}
.canvas-hint strong { color: var(--fg2); font-weight: 500; }

.stroke-info {
  display: flex;
  gap: 1.5rem;
  width: 100%;
  margin-top: 0.25rem;
}
.stroke-stat {
  flex: 1;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  padding: 0.6rem 0.75rem;
  text-align: center;
}
.stroke-stat-val {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent2);
  font-family: 'JetBrains Mono', monospace;
}
.stroke-stat-lbl {
  font-size: 0.65rem;
  color: var(--fg3);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 0.15rem;
}

/* ── RESULTS ── */
.result-hero {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 1.25rem;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 1.25rem;
}
.result-symbol {
  width: 90px; height: 90px;
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(129,140,248,0.08));
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: var(--radius-sm);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 0.35rem;
  color: var(--fg);
  flex-shrink: 0;
  box-shadow: 0 0 24px rgba(99,102,241,0.12);
  padding: 0.5rem;
  overflow: hidden;
}
.result-symbol .katex-render {
  font-size: 2rem;
  line-height: 1;
  color: var(--fg);
}
.result-symbol .katex-render .katex { color: var(--fg); }
.result-symbol .latex-code {
  font-size: 0.6rem;
  font-family: 'JetBrains Mono', monospace;
  color: var(--fg3);
  text-align: center;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 86px;
}
.topk-katex .katex { font-size: 1em; color: var(--fg); }
.topk-katex { display: inline-flex; align-items: center; min-width: 2.8rem; }
.result-meta { flex: 1; min-width: 0; }
.result-label {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--fg);
  letter-spacing: -0.02em;
  line-height: 1.2;
  font-family: 'JetBrains Mono', monospace;
}
.result-conf-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
.result-conf-badge {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.2rem 0.6rem;
  border-radius: 9999px;
}
.conf-high { background: var(--green-bg); color: var(--green); border: 1px solid rgba(16,185,129,0.25); }
.conf-mid  { background: var(--accent-bg); color: var(--accent2); border: 1px solid rgba(99,102,241,0.25); }
.conf-low  { background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }
.result-pct {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--fg);
  font-family: 'JetBrains Mono', monospace;
}

/* Confidence bar */
.conf-bar-wrap { margin-top: 0.6rem; }
.conf-bar-bg {
  width: 100%;
  height: 5px;
  background: var(--border2);
  border-radius: 9999px;
  overflow: hidden;
}
.conf-bar-fill {
  height: 100%;
  border-radius: 9999px;
  transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}

/* Top-K */
.topk-header {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg3);
  margin-bottom: 0.6rem;
}
.topk-row {
  display: grid;
  grid-template-columns: 1.2rem 3.5rem 1fr 3rem;
  align-items: center;
  gap: 0.6rem;
  padding: 0.55rem 0;
}
.topk-row + .topk-row { border-top: 1px solid var(--border); }
.topk-rank { font-size: 0.7rem; font-weight: 600; color: var(--fg3); text-align: right; }
.topk-sym {
  font-size: 1rem;
  font-weight: 600;
  color: var(--fg);
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.topk-bar-bg { height: 5px; background: var(--border2); border-radius: 9999px; overflow: hidden; }
.topk-bar-fill { height: 100%; border-radius: 9999px; }
.topk-pct { font-size: 0.75rem; font-weight: 600; color: var(--fg2); text-align: right;
            font-family: 'JetBrains Mono', monospace; }
.topk-row.rank1 .topk-sym { color: var(--accent2); }

/* Empty state */
.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 280px;
  gap: 0.75rem;
  color: var(--fg3);
  text-align: center;
}
.empty-icon {
  width: 56px; height: 56px;
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem;
}
.empty p { font-size: 0.9rem; color: var(--fg2); }
.empty span { font-size: 0.78rem; color: var(--fg3); }
</style>
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

# ── State Initialization ──
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "canvas_history" not in st.session_state:
    st.session_state.canvas_history = []
if "last_json" not in st.session_state:
    st.session_state.last_json = None
if "current_initial_drawing" not in st.session_state:
    st.session_state.current_initial_drawing = None

model     = load_model()
class_map = load_class_map()

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess_canvas(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
    if np.max(gray) < 10:
        return None
    gray   = 255 - gray
    coords = cv2.findNonZero(255 - gray)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    pad = 20
    x = max(x-pad,0); y = max(y-pad,0)
    w = min(w+2*pad, gray.shape[1]-x)
    h = min(h+2*pad, gray.shape[0]-y)
    crop  = gray[y:y+h, x:x+w]
    side  = max(crop.shape)
    sq    = np.full((side,side), 255, dtype=np.uint8)
    dy,dx = (side-crop.shape[0])//2, (side-crop.shape[1])//2
    sq[dy:dy+crop.shape[0], dx:dx+crop.shape[1]] = crop
    rsz   = cv2.resize(sq, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (rsz.astype("float32")/255.0).reshape(1,IMG_SIZE,IMG_SIZE,1)

# KaTeX rendered via st.components.v1.html (see results section below)

# ── TOPBAR ────────────────────────────────────────────────────────────────────
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

# ── LAYOUT ────────────────────────────────────────────────────────────────────
st.markdown('<div class="content-grid">', unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

# ── LEFT: CANVAS ──────────────────────────────────────────────────────────────
with col_l:
    st.markdown('<div id="left-panel-marker" style="display:none;"></div>', unsafe_allow_html=True)
    
    # ── Header Row ──
    h_col1, h_col2 = st.columns([1, 1])
    with h_col1:
        st.markdown('<div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--fg2); margin-top: 0.5rem;">Área de dibujo</div>', unsafe_allow_html=True)
    with h_col2:
        # Usamos columnas nativas para alinear los botones a la derecha horizontalmente
        btn_spacer, btn_undo, btn_clear = st.columns([2, 1, 1])
        with btn_undo:
            if st.button("↺", help="Deshacer", key="undo_btn", use_container_width=True):
                if len(st.session_state.canvas_history) > 0:
                    st.session_state.canvas_history.pop()
                    st.session_state.current_initial_drawing = st.session_state.canvas_history[-1] if st.session_state.canvas_history else None
                    st.session_state.last_json = st.session_state.current_initial_drawing
                    st.session_state.canvas_key += 1
                    st.rerun()
        with btn_clear:
            if st.button("🗑", help="Limpiar", key="clear_btn", use_container_width=True):
                st.session_state.canvas_history = []
                st.session_state.current_initial_drawing = None
                st.session_state.last_json = None
                st.session_state.canvas_key += 1
                st.rerun()

    # ── Canvas ──
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=14,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=320,
        width=320,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        initial_drawing=st.session_state.current_initial_drawing,
        display_toolbar=False,
    )

    # History tracking
    if canvas_result.json_data is not None:
        current_json = canvas_result.json_data
        if current_json != st.session_state.last_json:
            if current_json["objects"]:
                st.session_state.canvas_history.append(current_json)
            st.session_state.last_json = current_json

    st.markdown("""
          <div style="padding: 1.5rem;">
          <p class="canvas-hint">
            <strong>Dibuja</strong> un símbolo matemático con el ratón o dedo.<br>
            El modelo lo clasificará en tiempo real.
          </p>
          <div class="stroke-info">
            <div class="stroke-stat">
              <div class="stroke-stat-val">369</div>
              <div class="stroke-stat-lbl">Clases</div>
            </div>
            <div class="stroke-stat">
              <div class="stroke-stat-val">32px</div>
              <div class="stroke-stat-lbl">Resolución</div>
            </div>
            <div class="stroke-stat">
              <div class="stroke-stat-val">Top‑5</div>
              <div class="stroke-stat-lbl">Predicciones</div>
            </div>
          </div>
          </div>
    """, unsafe_allow_html=True)

# ── RIGHT: RESULTS ────────────────────────────────────────────────────────────
with col_r:
    # ── Build the results component HTML (runs in real iframe → KaTeX CDN works) ──
    has_result = False
    results_html = ""
    topk_rows_html = ""

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

            if best_conf >= 0.80:
                badge_cls = "conf-high"; badge_txt = "Alta confianza"
                bar_color = "#10B981"
            elif best_conf >= 0.50:
                badge_cls = "conf-mid"; badge_txt = "Confianza media"
                bar_color = "#6366F1"
            else:
                badge_cls = "conf-low"; badge_txt = "Confianza baja"
                bar_color = "#F59E0B"

            lbl_esc = best_label.replace('\\', '\\\\').replace('`', '\\`')

            # Build list of (element_id, latex) to render via KaTeX
            renders = [("sym-main", lbl_esc, False), ("sym-label", lbl_esc, False)]

            for rank, idx in enumerate(top_idxs, 1):
                lbl  = class_map.get(idx, "?")
                conf = float(preds[0][idx])
                rp   = int(conf * 100)
                bar  = "#6366F1" if rank == 1 else "#4B6080"
                cls  = "rank1" if rank == 1 else ""
                l    = lbl.replace('\\', '\\\\').replace('`', '\\`')
                renders.append((f"tksym{rank}", l, False))
                topk_rows_html += f"""
                <div class="topk-row {cls}">
                  <span class="topk-rank">{rank}</span>
                  <span class="topk-sym" id="tksym{rank}"></span>
                  <div class="topk-bar-bg">
                    <div class="topk-bar-fill" style="width:{rp}%;background:{bar};"></div>
                  </div>
                  <span class="topk-pct">{conf:.0%}</span>
                </div>
                """

            # Build render JS: one call per element, executed after KaTeX loads
            render_calls = "\n".join(
                f'  katex.render(`{lat}`, document.getElementById("{eid}"), {{throwOnError:false, displayMode:{str(dm).lower()}}});'
                for eid, lat, dm in renders
            )

            results_html = f"""
            <div class="result-hero">
              <div class="result-symbol">
                <div id="sym-main"></div>
                <span class="latex-code">{best_label}</span>
              </div>
              <div class="result-meta">
                <div class="result-label" id="sym-label"></div>
                <div class="result-conf-row">
                  <span class="result-conf-badge {badge_cls}">{badge_txt}</span>
                  <span class="result-pct">{best_conf:.1%}</span>
                </div>
                <div class="conf-bar-wrap">
                  <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
                  </div>
                </div>
              </div>
            </div>
            <div class="topk-header">Top {TOP_K} candidatos</div>
            {topk_rows_html}
            <script>
            window.addEventListener('load', function() {{
{render_calls}
            }});
            </script>
            """
    
    if not has_result:
        results_html = """
        <div class="empty">
          <div class="empty-icon">∫</div>
          <p>Dibuja un símbolo para comenzar</p>
          <span>El modelo CNN analizará tu trazo al instante</span>
        </div>
        """

    component_html = f"""
    <!DOCTYPE html><html><head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
    <style>
      * {{ box-sizing: border-box; margin:0; padding:0; }}
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
      body {{ background: transparent; font-family: 'Inter', sans-serif; color: #F0F4FF; }}

      .panel {{ background:#111827; border:1px solid #1E2D42; border-radius:18px;
                overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,.4); }}
      .panel-header {{ display:flex; align-items:center; justify-content:space-between;
                       padding:1rem 1.4rem; border-bottom:1px solid #1E2D42; background:#0D1421; }}
      .panel-title {{ font-size:.8rem; font-weight:600; text-transform:uppercase;
                      letter-spacing:.08em; color:#94A3B8; }}

      .panel-body {{ padding:1.5rem; }}

      .result-hero {{ display:flex; align-items:center; gap:1.5rem; padding:1.25rem;
                      background:#0D1421; border:1px solid #1E2D42; border-radius:12px;
                      margin-bottom:1.25rem; }}
      .result-symbol {{ width:90px; height:90px;
        background:linear-gradient(135deg,rgba(99,102,241,.15),rgba(129,140,248,.08));
        border:1px solid rgba(99,102,241,.25); border-radius:12px;
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:.3rem; flex-shrink:0; box-shadow:0 0 24px rgba(99,102,241,.12); padding:.5rem; overflow:hidden; }}
      .result-symbol .katex {{ font-size:2rem; color:#F0F4FF; }}
      .latex-code {{ font-size:.6rem; font-family:'JetBrains Mono',monospace; color:#4B6080;
                     text-align:center; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:86px; }}
      .result-meta {{ flex:1; min-width:0; }}
      .result-label .katex {{ font-size:1.35rem; font-weight:700; color:#F0F4FF; }}
      .result-conf-row {{ display:flex; align-items:center; gap:.5rem; margin-top:.5rem; }}
      .result-conf-badge {{ font-size:.72rem; font-weight:600; padding:.2rem .6rem; border-radius:9999px; }}
      .conf-high {{ background:rgba(16,185,129,.12); color:#10B981; border:1px solid rgba(16,185,129,.25); }}
      .conf-mid  {{ background:rgba(99,102,241,.12); color:#818CF8; border:1px solid rgba(99,102,241,.25); }}
      .conf-low  {{ background:rgba(245,158,11,.12); color:#F59E0B; border:1px solid rgba(245,158,11,.25); }}
      .result-pct {{ font-size:.85rem; font-weight:700; color:#F0F4FF; font-family:'JetBrains Mono',monospace; }}
      .conf-bar-wrap {{ margin-top:.6rem; }}
      .conf-bar-bg {{ width:100%; height:5px; background:#243347; border-radius:9999px; overflow:hidden; }}
      .conf-bar-fill {{ height:100%; border-radius:9999px; }}

      .topk-header {{ font-size:.7rem; font-weight:600; text-transform:uppercase;
                      letter-spacing:.08em; color:#4B6080; margin-bottom:.6rem; }}
      .topk-row {{ display:grid; grid-template-columns:1.2rem 3.5rem 1fr 3rem;
                   align-items:center; gap:.6rem; padding:.55rem 0; }}
      .topk-row + .topk-row {{ border-top:1px solid #1E2D42; }}
      .topk-rank {{ font-size:.7rem; font-weight:600; color:#4B6080; text-align:right; }}
      .topk-sym {{ font-size:1rem; font-weight:600; color:#F0F4FF; white-space:nowrap;
                   overflow:hidden; text-overflow:ellipsis; }}
      .topk-sym .katex {{ font-size:1rem; color:#F0F4FF; }}
      .rank1 .topk-sym .katex {{ color:#818CF8; }}
      .topk-bar-bg {{ height:5px; background:#243347; border-radius:9999px; overflow:hidden; }}
      .topk-bar-fill {{ height:100%; border-radius:9999px; }}
      .topk-pct {{ font-size:.75rem; font-weight:600; color:#94A3B8; text-align:right;
                   font-family:'JetBrains Mono',monospace; }}

      .empty {{ display:flex; flex-direction:column; align-items:center; justify-content:center;
                min-height:280px; gap:.75rem; color:#4B6080; text-align:center; }}
      .empty-icon {{ width:56px; height:56px; background:#1A2332; border:1px solid #243347;
                     border-radius:12px; display:flex; align-items:center; justify-content:center;
                     font-size:1.5rem; color:#94A3B8; }}
      .empty p {{ font-size:.9rem; color:#94A3B8; }}
      .empty span {{ font-size:.78rem; color:#4B6080; }}
    </style>
    </head><body>
    <div class="panel">
      <div class="panel-header"><span class="panel-title">Resultado del modelo</span></div>
      <div class="panel-body">
        {results_html}
      </div>
    </div>
    </body></html>
    """

    components.html(component_html, height=520, scrolling=False)

st.markdown("</div>", unsafe_allow_html=True)
