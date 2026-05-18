#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
  HASYv2 – Entrenamiento Profesional de Red Neuronal Convolucional
====================================================================
  Dataset:  HASYv2 (Handwritten Symbol Recognition)
  Salida:   modelo_hasy.keras  +  class_map.json
  Autor:    Generado automáticamente
  Stack:    TensorFlow 2.21 / Keras 3.14 / Python 3.13
====================================================================
"""

import os
import csv
import json
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silenciar warnings de TF

import tensorflow as tf
# pyrefly: ignore [missing-import]
import keras
# pyrefly: ignore [missing-import]
from keras import layers, callbacks, optimizers

# ──────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────
IMG_SIZE = 32
NUM_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 50          # Early stopping frenará si no mejora
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1  # 10% del train para validación

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "classification-task", "fold-1", "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "classification-task", "fold-1", "test.csv")
SYMBOLS_CSV = os.path.join(BASE_DIR, "symbols.csv")

# Salida directa a las carpetas del frontend (donde app.py los espera)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(FRONTEND_DIR, "keras", "modelo_hasy.keras")
CLASS_MAP_PATH = os.path.join(FRONTEND_DIR, "maps", "class_map.json")

# Crear directorios de salida si no existen
os.makedirs(os.path.join(FRONTEND_DIR, "keras"), exist_ok=True)
os.makedirs(os.path.join(FRONTEND_DIR, "maps"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# 2. CARGA DE DATOS (sin depender de hasy_tools.py)
# ──────────────────────────────────────────────────────────────────
def load_symbol_map(symbols_csv_path: str) -> dict:
    """
    Crea un mapeo bidireccional:
      symbol_id (str) → { 'index': int, 'latex': str }
    """
    symbol_map = {}
    with open(symbols_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            symbol_map[row["symbol_id"]] = {
                "index": idx,
                "latex": row["latex"],
            }
    return symbol_map


def load_dataset(csv_path: str, symbol_map: dict) -> tuple:
    """
    Lee un CSV del dataset y devuelve (images, labels) como numpy arrays.
    - images: float32, shape (N, 32, 32, 1), rango [0, 1]
    - labels: int32, shape (N,)
    """
    csv_dir = os.path.dirname(csv_path)
    paths, labels = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.normpath(os.path.join(csv_dir, row["path"]))
            sid = row["symbol_id"]
            if sid in symbol_map:
                paths.append(img_path)
                labels.append(symbol_map[sid]["index"])

    n = len(paths)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=np.float32)

    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE))
        images[i, :, :, 0] = np.array(img, dtype=np.float32) / 255.0

    return images, np.array(labels, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────
# 3. MODELO CNN OPTIMIZADO
# ──────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> keras.Model:
    """
    Arquitectura CNN profesional con:
    - Bloques Conv → BatchNorm → ReLU → MaxPool → Dropout
    - Capa densa con regularización
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="input_image")

    # Bloque 1: 32 filtros
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1a")(inputs)
    x = layers.BatchNormalization(name="bn1a")(x)
    x = layers.ReLU(name="relu1a")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1b")(x)
    x = layers.BatchNormalization(name="bn1b")(x)
    x = layers.ReLU(name="relu1b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Bloque 2: 64 filtros
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2a")(x)
    x = layers.BatchNormalization(name="bn2a")(x)
    x = layers.ReLU(name="relu2a")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2b")(x)
    x = layers.BatchNormalization(name="bn2b")(x)
    x = layers.ReLU(name="relu2b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Bloque 3: 128 filtros
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3a")(x)
    x = layers.BatchNormalization(name="bn3a")(x)
    x = layers.ReLU(name="relu3a")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3b")(x)
    x = layers.BatchNormalization(name="bn3b")(x)
    x = layers.ReLU(name="relu3b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Clasificador
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(0.5, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="HASYv2_CNN")
    return model


# ──────────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  HASYv2 – Entrenamiento de Modelo de Reconocimiento de Símbolos")
    print("=" * 65)

    # ── Cargar mapeo de símbolos ──
    print("\n📂 Cargando mapeo de símbolos...")
    symbol_map = load_symbol_map(SYMBOLS_CSV)
    num_classes = len(symbol_map)
    print(f"   Clases encontradas: {num_classes}")

    # ── Cargar datos ──
    print("\n📂 Cargando imágenes de entrenamiento...")
    t0 = time.time()
    x_train_full, y_train_full = load_dataset(TRAIN_CSV, symbol_map)
    print(f"   {len(x_train_full):,} imágenes cargadas en {time.time()-t0:.1f}s")

    print("\n📂 Cargando imágenes de test...")
    t0 = time.time()
    x_test, y_test = load_dataset(TEST_CSV, symbol_map)
    print(f"   {len(x_test):,} imágenes cargadas en {time.time()-t0:.1f}s")

    # ── Split train / validation ──
    n_val = int(len(x_train_full) * VALIDATION_SPLIT)
    indices = np.random.permutation(len(x_train_full))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]
    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]

    print(f"\n📊 Distribución de datos:")
    print(f"   Entrenamiento: {len(x_train):>7,} imágenes")
    print(f"   Validación:    {len(x_val):>7,} imágenes")
    print(f"   Test:          {len(x_test):>7,} imágenes")

    # ── Data Augmentation ──
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.08, fill_mode="constant", fill_value=1.0, name="aug_rotation"),
        layers.RandomZoom((-0.1, 0.1), fill_mode="constant", fill_value=1.0, name="aug_zoom"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=1.0, name="aug_translate"),
    ], name="data_augmentation")

    # ── Crear datasets tf.data (rendimiento óptimo) ──
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE)
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ── Construir modelo ──
    print("\n🏗️  Construyendo modelo CNN...")
    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ──
    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Entrenar ──
    print("\n🚀 Iniciando entrenamiento...")
    print(f"   Epochs máximos: {EPOCHS}")
    print(f"   Batch size:     {BATCH_SIZE}")
    print(f"   Learning rate:  {LEARNING_RATE}")
    print("-" * 65)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1,
    )

    # ── Evaluación final ──
    print("\n" + "=" * 65)
    print("📈 Evaluación en el conjunto de TEST:")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"   Loss:     {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # ── Guardar modelo final (por si el checkpoint no se disparó al final) ──
    model.save(MODEL_PATH)
    print(f"\n💾 Modelo guardado en: {MODEL_PATH}")

    # ── Guardar mapeo de clases (necesario para la app web) ──
    class_map_export = {}
    for sid, info in symbol_map.items():
        class_map_export[str(info["index"])] = {
            "symbol_id": sid,
            "latex": info["latex"],
        }

    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(class_map_export, f, ensure_ascii=False, indent=2)
    print(f"📋 Mapeo de clases guardado en: {CLASS_MAP_PATH}")

    # ── Resumen ──
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"\n{'=' * 65}")
    print(f"  ✅ ENTRENAMIENTO COMPLETADO")
    print(f"     Mejor val_accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"     Test accuracy:      {test_acc:.4f}")
    print(f"     Archivo modelo:     {os.path.basename(MODEL_PATH)}")
    print(f"     Archivo clases:     {os.path.basename(CLASS_MAP_PATH)}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
