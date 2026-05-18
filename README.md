# ∑ HASYv2 – Reconocedor de Símbolos Matemáticos

Aplicación web que reconoce **369 símbolos matemáticos** escritos a mano en tiempo real, usando una red neuronal convolucional entrenada sobre el dataset [HASYv2](https://zenodo.org/record/259444).

Dibujas un símbolo en el canvas, el modelo lo clasifica al instante y te devuelve las 5 predicciones más probables con su porcentaje de confianza.

---

## Qué hay dentro

El proyecto se divide en dos partes bien diferenciadas:

- **Entrenamiento** (`entrenamiento.py`) — Script que carga las ~168.000 imágenes del dataset, entrena un modelo CNN desde cero y exporta los artefactos necesarios (`.keras` + `class_map.json`).
- **Frontend** (`frontend/`) — Interfaz web construida con Streamlit que carga el modelo entrenado y permite dibujar símbolos para clasificarlos en tiempo real.

### Stack técnico

| Capa | Tecnología |
|---|---|
| Modelo | TensorFlow 2.x / Keras |
| Frontend | Streamlit + streamlit-drawable-canvas |
| Procesamiento de imagen | OpenCV (headless) + NumPy + Pillow |
| Dataset | HASYv2 (168.236 imágenes, 32×32 px, 369 clases) |

---

## Requisitos previos

- **Python 3.10+** (probado con 3.13)
- **pip** actualizado
- Recomendado: usar un entorno virtual para no contaminar el sistema

---

## Instalación

1. **Clonar el repositorio**

```bash
git clone <url-del-repo>
cd Frontend-HasyV2
```

2. **Crear entorno virtual** (opcional pero recomendado)

```bash
python3 -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

Esto instalará: `streamlit`, `tensorflow`, `streamlit-drawable-canvas`, `opencv-python-headless` y `numpy`.

> **Nota sobre Pillow:** Se usa internamente para el entrenamiento. Si no la tienes instalada, `pip install Pillow`.

---

## Cómo entrenar el modelo

Si quieres entrenar el modelo desde cero (o reentrenarlo con otros parámetros), necesitas tener el dataset HASYv2 descomprimido en la raíz del proyecto. La estructura esperada es:

```
Frontend-HasyV2/
├── hasy-data/            ← ~168.000 imágenes PNG (32×32)
├── hasy-data-labels.csv
├── classification-task/
│   ├── fold-1/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── fold-2/
│   │   └── ...
│   └── fold-10/
├── symbols.csv
└── entrenamiento.py
```

Para lanzar el entrenamiento:

```bash
python entrenamiento.py
```

### Qué hace el script

1. Carga el mapeo de las 369 clases desde `symbols.csv`.
2. Lee las imágenes de entrenamiento y test usando los CSVs del fold-1.
3. Separa un 10% del train como validación.
4. Aplica data augmentation (rotación, zoom, traslación) al vuelo.
5. Entrena una CNN con 3 bloques convolucionales (32 → 64 → 128 filtros) + BatchNorm + Dropout.
6. Usa Early Stopping (paciencia 7 epochs) y ReduceLROnPlateau para optimizar el aprendizaje.
7. Guarda el mejor modelo en `modelo_hasy.keras` y el mapeo de clases en `class_map.json`.

### Parámetros del entrenamiento

Se pueden modificar directamente en las primeras líneas de `entrenamiento.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `IMG_SIZE` | 32 | Tamaño de entrada (px) |
| `BATCH_SIZE` | 128 | Tamaño del batch |
| `EPOCHS` | 50 | Epochs máximos (early stopping frenará antes) |
| `LEARNING_RATE` | 1e-3 | Learning rate inicial (Adam) |
| `VALIDATION_SPLIT` | 0.1 | Fracción del train usada para validación |

### Salida del entrenamiento

Una vez termine, se generan dos archivos directamente en las carpetas del frontend:

- `frontend/keras/modelo_hasy.keras` — Modelo entrenado listo para inferencia.
- `frontend/maps/class_map.json` — Diccionario que mapea cada índice de clase a su `symbol_id` y código LaTeX.

No hace falta copiar nada manualmente: el script ya los guarda donde `app.py` los espera.

> El repositorio ya incluye un modelo preentrenado listo para usar, así que solo necesitas reentrenar si quieres ajustar los parámetros o experimentar.

---

## Cómo ejecutar la aplicación

```bash
cd frontend
streamlit run app.py
```

Se abrirá automáticamente en el navegador (por defecto `http://localhost:8501`).

### Uso

1. **Dibuja** un símbolo matemático en el canvas negro con el ratón o el dedo (en dispositivos táctiles).
2. El modelo **clasifica** el trazo en tiempo real y muestra el resultado con un indicador de confianza.
3. Puedes ver el **Top 5** de predicciones con sus porcentajes.
4. Botón **↺** para deshacer el último trazo, **🗑** para limpiar todo el canvas.
5. Pulsa **"369 clases"** en la barra superior para explorar todos los símbolos que reconoce el modelo, con buscador incluido.

### Categorías de símbolos soportados

El modelo reconoce un abanico bastante amplio de símbolos:

- **Letras** — Mayúsculas (A-Z), minúsculas (a-z)
- **Números** — 0-9
- **Letras griegas** — α, β, γ, δ, Σ, Ω, etc.
- **Operadores** — +, −, ×, ÷, ±, ⊕, ⊗, etc.
- **Relaciones** — =, ≠, ≤, ≥, ≈, ≡, ⊂, ∈, etc.
- **Flechas** — →, ⇒, ↔, ⟹, ↦, etc.
- **Integrales y sumatorios** — ∫, ∮, ∑, ∏, ∂, ∇
- **Lógica y conjuntos** — ∀, ∃, ∪, ∩, ∧, ∨, ¬
- **Notación especial** — ℝ, ℕ, ℤ, ℚ, ∞, ℵ, √, etc.
- **Misceláneos** — ♡, ♣, ©, §, °, ℃, etc.

---

## Estructura del proyecto

```
Frontend-HasyV2/
├── entrenamiento.py           # Script de entrenamiento del modelo CNN
├── hasy_tools.py              # Herramientas auxiliares del dataset HASYv2
├── requirements.txt           # Dependencias Python
├── symbols.csv                # Listado de las 369 clases con su código LaTeX
├── hasy-data-labels.csv       # Labels de todas las imágenes
├── hasy-data/                 # Imágenes del dataset (168.236 PNGs, 32×32)
├── classification-task/       # 10 folds para cross-validation
│   ├── fold-1/
│   │   ├── train.csv
│   │   └── test.csv
│   └── ...
├── verification-task/         # Datos para la tarea de verificación de pares
│   ├── train.csv
│   ├── test-v1.csv
│   ├── test-v2.csv
│   └── test-v3.csv
├── frontend/
│   ├── app.py                 # Aplicación Streamlit (interfaz principal)
│   ├── keras/
│   │   └── modelo_hasy.keras  # Modelo CNN preentrenado (~11 MB)
│   ├── maps/
│   │   └── class_map.json     # Mapeo clase → LaTeX
│   └── styles/
│       └── main.css           # Estilos CSS de la interfaz
└── .gitignore
```

---

## Arquitectura del modelo

La CNN sigue una arquitectura clásica pensada para imágenes pequeñas (32×32):

```
Input (32×32×1)
    │
    ├─ Bloque 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
    ├─ Bloque 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
    ├─ Bloque 3: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25)
    │
    ├─ Flatten → Dense(256) → BN → ReLU → Dropout(0.5)
    └─ Dense(369, softmax) → Predicción
```

Cada bloque convolucional duplica el número de filtros respecto al anterior, y el dropout progresivo ayuda a evitar sobreajuste con un dataset que tiene clases bastante desbalanceadas (algunas tienen 3.000+ muestras y otras apenas 45).

---

## Notas adicionales

- **hasy_tools.py** es un módulo auxiliar del dataset original (de Martin Thoma). Incluye funciones para análisis de datos, PCA, distribución de clases, etc. No se usa directamente en la app web, pero es útil si quieres hacer análisis exploratorio del dataset.

- **Preprocesamiento en inferencia**: cuando dibujas en el canvas, la imagen pasa por un pipeline que la convierte a escala de grises, invierte los colores, recorta la región con contenido, la centra en un cuadrado y la redimensiona a 32×32 px antes de pasarla al modelo.

- **Rendimiento**: el modelo se carga una sola vez en memoria (usando `@st.cache_resource`), así que la inferencia es prácticamente instantánea después de la primera carga.

---

## Sobre el dataset

El dataset HASYv2 fue creado por [Martin Thoma](https://martin-thoma.com/) y contiene 168.236 imágenes de símbolos manuscritos distribuidas en 369 clases. Cada imagen es un PNG de 32×32 píxeles en escala de grises.

Para más detalles sobre el dataset: [paper original](https://arxiv.org/abs/1701.08380) | [repositorio](https://github.com/MartinThoma/HASY)

---

## Licencia

El dataset HASYv2 se distribuye bajo su propia licencia (ver el paper original). El código de este proyecto es de uso libre.
