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

# ── CSS global (cargado desde styles.css) ────────────────────────────────────
def _inject_css():
    css_path = os.path.join(BASE_DIR, "styles.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_inject_css()

# ── KaTeX (CDN) ───────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
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
    ("show_classes", False),
    ("class_search", ""),
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
LATEX_UNICODE: dict = {
    # Greek lowercase
    r"\alpha":"α", r"\beta":"β", r"\gamma":"γ", r"\delta":"δ",
    r"\epsilon":"ε", r"\varepsilon":"ε", r"\zeta":"ζ", r"\eta":"η",
    r"\theta":"θ", r"\vartheta":"ϑ", r"\iota":"ι", r"\kappa":"κ",
    r"\varkappa":"ϰ", r"\lambda":"λ", r"\mu":"μ", r"\nu":"ν",
    r"\xi":"ξ", r"\pi":"π", r"\rho":"ρ", r"\varrho":"ϱ",
    r"\sigma":"σ", r"\varsigma":"ς", r"\tau":"τ",
    r"\phi":"φ", r"\varphi":"φ", r"\chi":"χ", r"\psi":"ψ", r"\omega":"ω",
    # Greek uppercase
    r"\Gamma":"Γ", r"\Delta":"Δ", r"\Theta":"Θ", r"\Lambda":"Λ",
    r"\Xi":"Ξ", r"\Pi":"Π", r"\Sigma":"Σ", r"\Phi":"Φ",
    r"\Psi":"Ψ", r"\Omega":"Ω",
    # Arrows
    r"\rightarrow":"→", r"\leftarrow":"←", r"\uparrow":"↑", r"\downarrow":"↓",
    r"\Rightarrow":"⇒", r"\Leftarrow":"⇐", r"\Uparrow":"⇑", r"\Downarrow":"⇓",
    r"\leftrightarrow":"↔", r"\Leftrightarrow":"⇔",
    r"\longrightarrow":"⟶", r"\longleftarrow":"⟵",
    r"\Longrightarrow":"⟹", r"\Longleftarrow":"⟸",
    r"\Longleftrightarrow":"⟺", r"\longleftrightarrow":"⟷",
    r"\mapsto":"↦", r"\longmapsto":"⟼",
    r"\nearrow":"↗", r"\searrow":"↘", r"\swarrow":"↙", r"\nwarrow":"↖",
    r"\hookrightarrow":"↪", r"\hookleftarrow":"↩",
    r"\twoheadrightarrow":"↠", r"\twoheadleftarrow":"↞",
    r"\rightsquigarrow":"⇝", r"\leftrightharpoons":"⇋",
    r"\rightleftharpoons":"⇌", r"\multimap":"⊸",
    r"\nrightarrow":"↛", r"\nleftarrow":"↚",
    r"\nRightarrow":"⇏", r"\nLeftarrow":"⇍",
    r"\upharpoonright":"↾", r"\downharpoonright":"⇂",
    # Large operators
    r"\sum":"∑", r"\prod":"∏", r"\coprod":"∐",
    r"\int":"∫", r"\oint":"∮", r"\iint":"∬", r"\iiint":"∭",
    r"\oiint":"∯", r"\fint":"⨏", r"\varoiint":"∯",
    r"\partial":"∂", r"\nabla":"∇",
    # Binary operations
    r"\times":"×", r"\div":"÷", r"\pm":"±", r"\mp":"∓",
    r"\cdot":"·", r"\bullet":"•", r"\circ":"∘", r"\star":"⋆",
    r"\ast":"∗", r"\oplus":"⊕", r"\ominus":"⊖", r"\otimes":"⊗",
    r"\oslash":"⊘", r"\odot":"⊙", r"\circledcirc":"⊚",
    r"\circledast":"⊛", r"\diamond":"⋄",
    r"\triangleleft":"◁", r"\triangleright":"▷",
    r"\cup":"∪", r"\cap":"∩", r"\sqcup":"⊔", r"\sqcap":"⊓",
    r"\vee":"∨", r"\wedge":"∧", r"\setminus":"∖",
    r"\uplus":"⊎", r"\amalg":"⨿",
    r"\boxplus":"⊞", r"\boxminus":"⊟", r"\boxtimes":"⊠", r"\boxdot":"⊡",
    r"\ltimes":"⋉", r"\rtimes":"⋊", r"\lhd":"⊲", r"\rhd":"⊳",
    r"\wr":"≀", r"\barwedge":"⊼", r"\curlywedge":"⋏", r"\curlyvee":"⋎",
    r"\parr":"⅋", r"\with":"&",
    # Relations
    r"\leq":"≤", r"\geq":"≥", r"\neq":"≠", r"\approx":"≈",
    r"\equiv":"≡", r"\not\equiv":"≢", r"\sim":"∼", r"\simeq":"≃",
    r"\cong":"≅", r"\propto":"∝", r"\varpropto":"∝",
    r"\perp":"⊥", r"\parallel":"∥", r"\mid":"∣",
    r"\subset":"⊂", r"\supset":"⊃", r"\subseteq":"⊆", r"\supseteq":"⊇",
    r"\sqsubset":"⊏", r"\sqsupset":"⊐",
    r"\sqsubseteq":"⊑", r"\sqsupseteq":"⊒",
    r"\in":"∈", r"\notin":"∉", r"\ni":"∋",
    r"\prec":"≺", r"\succ":"≻", r"\preceq":"⪯", r"\succeq":"⪰",
    r"\preccurlyeq":"≼", r"\succcurlyeq":"≽",
    r"\ll":"≪", r"\gg":"≫",
    r"\asymp":"≍", r"\bowtie":"⋈", r"\smile":"⌣", r"\frown":"⌢",
    r"\vdash":"⊢", r"\dashv":"⊣", r"\models":"⊨",
    r"\Vdash":"⊩", r"\vDash":"⊨",
    r"\doteq":"≐", r"\backsim":"∽", r"\pitchfork":"⋔",
    r"\between":"≬", r"\therefore":"∴", r"\because":"∵",
    r"\lesssim":"≲", r"\gtrsim":"≳",
    r"\lessgtr":"≶", r"\gtrless":"≷",
    r"\trianglelefteq":"⊴", r"\trianglerighteq":"⊵",
    r"\eqslantless":"⪕", r"\eqslantgtr":"⪖",
    r"\lesseqqgtr":"⪋", r"\gtreqqless":"⪌",
    # Misc math
    r"\infty":"∞", r"\forall":"∀", r"\exists":"∃", r"\nexists":"∄",
    r"\checkmark":"✓", r"\top":"⊤", r"\bot":"⊥",
    r"\angle":"∠", r"\measuredangle":"∡", r"\sphericalangle":"∢",
    r"\triangle":"△", r"\square":"□", r"\blacksquare":"■",
    r"\lozenge":"◊", r"\blacklozenge":"⧫",
    r"\blacktriangle":"▲", r"\blacktriangledown":"▼",
    r"\blacktriangleleft":"◀", r"\blacktriangleright":"▶",
    r"\triangledown":"▽",
    r"\aleph":"ℵ", r"\hbar":"ℏ", r"\ell":"ℓ", r"\wp":"℘",
    r"\emptyset":"∅", r"\varnothing":"∅",
    r"\dots":"…", r"\ldots":"…", r"\cdots":"⋯", r"\vdots":"⋮", r"\ddots":"⋱",
    r"\prime":"′",
    r"\heartsuit":"♡", r"\diamondsuit":"♢", r"\clubsuit":"♣", r"\spadesuit":"♠",
    r"\flat":"♭", r"\sharp":"♯", r"\natural":"♮",
    r"\sqrt{}":"√",
    # Special chars
    r"\$":"$", r"\{":"{", r"\}":"}", r"\#":"#", r"\%":"%",
    r"\&":"&", r"\S":"§", r"\dag":"†", r"\ddag":"‡",
    r"\pounds":"£", r"\copyright":"©", r"\circledR":"®",
    r"\mathsection":"§",
    # Brackets
    r"\langle":"⟨", r"\rangle":"⟩",
    r"\lceil":"⌈", r"\rceil":"⌉", r"\lfloor":"⌊", r"\rfloor":"⌋",
    r"\lbracket":"[", r"\rbracket":"]",
    r"\lvert":"|", r"\rvert":"|", r"\lVert":"‖", r"\rVert":"‖",
    # Mathbb
    r"\mathbb{R}":"ℝ", r"\mathbb{N}":"ℕ", r"\mathbb{Z}":"ℤ",
    r"\mathbb{Q}":"ℚ", r"\mathbb{C}":"ℂ",
    r"\mathbb{b}":"𝕓", r"\mathbb{0}":"𝟘", r"\mathbb{1}":"𝟙",
    # Mathds (double-struck / blackboard bold)
    r"\mathds{1}":"𝟙", r"\mathds{C}":"ℂ", r"\mathds{E}":"𝔼",
    r"\mathds{N}":"ℕ", r"\mathds{P}":"ℙ", r"\mathds{Q}":"ℚ",
    r"\mathds{R}":"ℝ", r"\mathds{Z}":"ℤ",
    # Mathscr (script)
    r"\mathscr{A}":"𝒜", r"\mathscr{C}":"𝒞", r"\mathscr{D}":"𝒟",
    r"\mathscr{E}":"ℰ", r"\mathscr{F}":"ℱ", r"\mathscr{H}":"ℋ",
    r"\mathscr{L}":"ℒ", r"\mathscr{P}":"𝒫", r"\mathscr{S}":"𝒮",
    # Mathcal
    r"\mathcal{A}":"𝒜", r"\mathcal{B}":"ℬ", r"\mathcal{C}":"𝒞",
    r"\mathcal{D}":"𝒟", r"\mathcal{E}":"ℰ", r"\mathcal{F}":"ℱ",
    r"\mathcal{G}":"𝒢", r"\mathcal{H}":"ℋ", r"\mathcal{I}":"ℐ",
    r"\mathcal{J}":"𝒥", r"\mathcal{K}":"𝒦", r"\mathcal{L}":"ℒ",
    r"\mathcal{M}":"ℳ", r"\mathcal{N}":"𝒩", r"\mathcal{O}":"𝒪",
    r"\mathcal{P}":"𝒫", r"\mathcal{Q}":"𝒬", r"\mathcal{R}":"ℛ",
    r"\mathcal{S}":"𝒮", r"\mathcal{T}":"𝒯", r"\mathcal{U}":"𝒰",
    r"\mathcal{V}":"𝒱", r"\mathcal{W}":"𝒲", r"\mathcal{X}":"𝒳",
    r"\mathcal{Y}":"𝒴", r"\mathcal{Z}":"𝒵",
    # Mathfrak
    r"\mathfrak{A}":"𝔄", r"\mathfrak{B}":"𝔅", r"\mathfrak{C}":"ℭ",
    r"\mathfrak{D}":"𝔇", r"\mathfrak{E}":"𝔈", r"\mathfrak{F}":"𝔉",
    r"\mathfrak{G}":"𝔊", r"\mathfrak{H}":"ℌ", r"\mathfrak{I}":"ℑ",
    r"\mathfrak{J}":"𝔍", r"\mathfrak{K}":"𝔎", r"\mathfrak{L}":"𝔏",
    r"\mathfrak{M}":"𝔐", r"\mathfrak{N}":"𝔑", r"\mathfrak{O}":"𝔒",
    r"\mathfrak{P}":"𝔓", r"\mathfrak{Q}":"𝔔", r"\mathfrak{R}":"ℜ",
    r"\mathfrak{S}":"𝔖", r"\mathfrak{T}":"𝔗", r"\mathfrak{U}":"𝔘",
    r"\mathfrak{V}":"𝔙", r"\mathfrak{W}":"𝔚", r"\mathfrak{X}":"𝔛",
    r"\mathfrak{Y}":"𝔜", r"\mathfrak{Z}":"ℨ",
    r"\mathfrak{a}":"𝔞", r"\mathfrak{b}":"𝔟", r"\mathfrak{c}":"𝔠",
    r"\mathfrak{d}":"𝔡", r"\mathfrak{e}":"𝔢", r"\mathfrak{f}":"𝔣",
    r"\mathfrak{g}":"𝔤", r"\mathfrak{h}":"𝔥", r"\mathfrak{i}":"𝔦",
    r"\mathfrak{j}":"𝔧", r"\mathfrak{k}":"𝔨", r"\mathfrak{l}":"𝔩",
    r"\mathfrak{m}":"𝔪", r"\mathfrak{n}":"𝔫", r"\mathfrak{o}":"𝔬",
    r"\mathfrak{p}":"𝔭", r"\mathfrak{q}":"𝔮", r"\mathfrak{r}":"𝔯",
    r"\mathfrak{s}":"𝔰", r"\mathfrak{t}":"𝔱", r"\mathfrak{u}":"𝔲",
    r"\mathfrak{v}":"𝔳", r"\mathfrak{w}":"𝔴", r"\mathfrak{x}":"𝔵",
    r"\mathfrak{y}":"𝔶", r"\mathfrak{z}":"𝔷",
    # Mathbb lowercase/digits
    r"\mathbb{a}":"𝕒", r"\mathbb{b}":"𝕓", r"\mathbb{c}":"𝕔",
    # Astronomy / misc
    r"\venus":"♀", r"\mars":"♂", r"\sun":"☉", r"\moon":"☽",
    r"\male":"♂", r"\female":"♀",
    r"\astrosun":"☉", r"\fullmoon":"○", r"\leftmoon":"☽",
    r"\backslash":"\\", r"\diameter":"⌀",
    r"\celsius":"℃", r"\ohm":"Ω", r"\degree":"°",
    r"\guillemotleft":"«", r"\guillemotright":"»",
    r"\AE":"Æ", r"\ae":"æ", r"\OE":"Œ", r"\oe":"œ",
    r"\aa":"å", r"\AA":"Å",
    r"\checked":"✓",
    r"\Im":"ℑ", r"\Re":"ℜ",
    r"\O":"Ø", r"\o":"ø",
    r"\L":"Ł", r"\l":"ł",
    r"\ss":"ß",
    # Extra arrows
    r"\rightharpoonup":"⇀", r"\leadsto":"⇝",
    r"\circlearrowleft":"↺", r"\circlearrowright":"↻",
    r"\rightleftarrows":"⇄", r"\rightrightarrows":"⇉",
    r"\mapsfrom":"↤", r"\shortrightarrow":"→",
    r"\lightning":"↯",
    # Extra math / relations
    r"\mathbb{H}":"ℍ",
    r"\Bowtie":"⋈", r"\vartriangle":"△",
    r"\iddots":"⋰",
    # Double brackets
    r"\llbracket":"⟦", r"\rrbracket":"⟧",
    # Dots
    r"\dotsc":"…",
    # Greek
    r"\varpi":"ϖ",
    # Extra arrows
    r"\curvearrowright":"↷",
    # Extra relations
    r"\triangleq":"≜",
    r"\leqslant":"⩽", r"\geqslant":"⩾",
    r"\nmid":"∤",
    r"\nvDash":"⊭", r"\nsubseteq":"⊈",
    r"\subsetneq":"⊊", r"\varsubsetneq":"⊊",
    # Logical / misc
    r"\neg":"¬", r"\lnot":"¬",
    r"\vee":"∨", r"\lor":"∨", r"\wedge":"∧", r"\land":"∧",
    r"\oplus":"⊕", r"\bigoplus":"⊕",
    r"\Browseq": "?",
}

def latex_to_display(latex: str) -> str:
    return LATEX_UNICODE.get(latex, latex)

# ── TOPBAR ───────────────────────────────────────────────────────────────────
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
    <span class="pill pill-mono">CNN · 32×32</span>
    <span class="pill pill-mono">TensorFlow</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Botón "369 clases" dentro del topbar via CSS absolute positioning
if st.button("📐 369 clases", key="btn_clases", help="Ver todas las clases disponibles"):
    st.session_state.show_classes = not st.session_state.show_classes
    st.rerun()

# ── CLASSES VIEW ─────────────────────────────────────────────────────────────
if st.session_state.show_classes:
    # Auto-scroll to top of page when entering classes view
    st.markdown("""
    <script>
    (function() {
        function scrollTop() {
            var main = window.parent.document.querySelector('section.main');
            if (main) main.scrollTop = 0;
            var body = window.parent.document.querySelector('[data-testid="stMainBlockContainer"]');
            if (body) body.scrollTop = 0;
            window.parent.scrollTo(0, 0);
        }
        scrollTop();
        setTimeout(scrollTop, 100);
    })();
    </script>
    """, unsafe_allow_html=True)
    @st.cache_data
    def load_class_map_full():
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    full_map = load_class_map_full()

    # Header + botón volver en HTML puro (sin st.columns para no romper CSS)
    st.markdown("""
    <div class="classes-header">
      <div>
        <h2>📐 Todas las clases &middot; HASYv2</h2>
        <p>El modelo CNN reconoce <strong style="color:var(--accent2)">369 símbolos matemáticos</strong> distintos</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Volver al clasificador", key="btn_back"):
        st.session_state.show_classes = False
        st.rerun()

    # Búsqueda
    search_q = st.text_input("", placeholder="🔍  Buscar símbolo LaTeX...", key="class_search",
                             label_visibility="collapsed")

    # Filtrar
    items = [(int(k), v["latex"]) for k, v in full_map.items()]
    items.sort(key=lambda x: x[0])
    if search_q:
        items = [(idx, ltx) for idx, ltx in items if search_q.lower() in ltx.lower()]

    # Grid: use Python LATEX_UNICODE dict — works for all 369 symbols without KaTeX
    cards_html = []
    for idx, ltx in items:
        glyph = latex_to_display(ltx)
        ltx_safe = ltx.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        glyph_safe = glyph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        cards_html.append(
            f'<div class="class-card">'
            f'<span class="class-card-id">#{idx}</span>'
            f'<span class="class-card-glyph">{glyph_safe}</span>'
            f'<span class="class-card-latex">{ltx_safe}</span>'
            f'</div>'
        )

    if not cards_html:
        cards_html.append('<div class="classes-empty">No se encontraron clases.</div>')

    n_items = len(items)
    # Fixed viewport height: the grid scrolls inside the component
    # Calculate a reasonable height: full viewport minus topbar (~56px) and controls (~110px)
    initial_h = 700

    component_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<style>
  :root {{
    --bg: #080C14; --surface: #111827; --border: #1E2D42; --border2: #243347;
    --surface2: #1A2332; --fg: #F0F4FF; --fg3: #475569; --fg4: #334155;
    --accent: #6366F1; --accent-bg: rgba(99,102,241,0.10);
    --radius-xs: 7px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; }}
  body {{ background: var(--bg); font-family: 'Inter', system-ui, sans-serif; padding: 0; margin: 0; overflow: hidden; }}
  .scroll-container {{
    height: 100vh;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 0.5rem 0 1rem;
    scrollbar-width: thin;
    scrollbar-color: #243347 #080C14;
  }}
  .scroll-container::-webkit-scrollbar {{ width: 6px; }}
  .scroll-container::-webkit-scrollbar-track {{ background: #080C14; }}
  .scroll-container::-webkit-scrollbar-thumb {{ background: #243347; border-radius: 3px; }}
  .scroll-container::-webkit-scrollbar-thumb:hover {{ background: #6366F1; }}
  .classes-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
    gap: 8px;
  }}
  .class-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xs);
    padding: 10px 8px;
    display: flex; flex-direction: column;
    align-items: center; gap: 5px;
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
  }}
  .class-card:hover {{
    border-color: var(--accent);
    background: var(--surface2);
    transform: translateY(-1px);
  }}
  .class-card-id {{
    font-size: 9px; color: var(--fg4);
    font-family: 'JetBrains Mono', monospace; line-height: 1;
  }}
  .class-card-glyph {{
    font-size: 20px; color: var(--fg);
    min-height: 28px;
    display: flex; align-items: center; justify-content: center;
  }}
  .class-card-glyph .katex {{ font-size: 1.1em !important; color: var(--fg) !important; }}
  .class-card-glyph .katex * {{ color: inherit !important; }}
  .class-card-latex {{
    font-size: 9px; color: var(--fg3);
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 80px;
  }}
  .classes-empty {{
    grid-column: 1/-1; text-align: center;
    padding: 3rem; color: var(--fg4); font-size: 13px;
  }}
</style>
</head>
<body>
<div class="scroll-container" id="scroll">
  <div class="classes-grid" id="grid">
  {"".join(cards_html)}
  </div>
</div>
<script>
// Set iframe height to fill remaining viewport
function setHeight() {{
  var viewportH = window.parent.innerHeight || 700;
  var topbar = 56;   // topbar height
  var controls = 115; // back button + search input
  var h = Math.max(viewportH - topbar - controls, 400);
  window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: h}}, '*');
  document.getElementById('scroll').style.height = h + 'px';
}}
setHeight();
window.addEventListener('resize', setHeight);
window.addEventListener('load', setHeight);
</script>
</body>
</html>
"""
    import streamlit.components.v1 as components
    components.html(component_html, height=initial_h, scrolling=False)
    st.stop()

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
