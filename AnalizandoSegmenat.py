# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:54:05 2026

@author: dkpin
"""

"""
DIAGNÓSTICO DE MÁSCARA — Bell Pepper
Ejecutar con: python diagnostico_mascara.py
Necesita: PepMaskMobilnet.tflite y una imagen de prueba
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
TFLITE_PATH = "PepMaskMobilnet.tflite"   # ajusta si está en otra ruta
IMG_PATH    = "D:/DATASETS/Imagenes/PlantVillage/Dataset/Plant_leave_diseases_dataset_without_augmentation/Pepper,_bell___Bacterial_spot/image (25).JPG"   # pon aquí la ruta a una imagen, o déjalo en None para imagen sintética
SIZE        = 256
NUM_CLASSES = 2
CLASS_NAMES = ["Bacterial Spot", "Healthy"]

# ── 1. CARGAR MODELO ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. INSPECCIONANDO MODELO TFLITE")
print("=" * 60)

interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()

inp  = interp.get_input_details()
outp = interp.get_output_details()

print(f"\nEntrada:")
print(f"  shape : {inp[0]['shape']}")
print(f"  dtype : {inp[0]['dtype']}")

print(f"\nSalidas ({len(outp)} tensores):")
for i, o in enumerate(outp):
    print(f"  [{i}] name={o['name']:<40} shape={o['shape']}  dtype={o['dtype']}")

# Identificar cuál es máscara y cuál es clase
mask_idx  = next(i for i, o in enumerate(outp) if len(o['shape']) == 4)
class_idx = next(i for i, o in enumerate(outp) if len(o['shape']) == 2)
print(f"\nmask_idx={mask_idx}  class_idx={class_idx}")

# ── 2. PREPARAR IMAGEN ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PREPARANDO IMAGEN")
print("=" * 60)

if IMG_PATH and os.path.exists(IMG_PATH):
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    print(f"  Imagen cargada: {IMG_PATH}")
else:
    # Imagen sintética: hoja verde con manchas marrones
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:, :] = [34, 139, 34]   # verde hoja
    # manchas
    cv2.circle(img, (80, 80),   20, (139, 69, 19), -1)
    cv2.circle(img, (180, 150), 15, (101, 67, 33), -1)
    cv2.circle(img, (120, 200), 25, (139, 69, 19), -1)
    print("  Usando imagen sintética (hoja verde con manchas)")

# Preprocesar igual que Android: MobileNetV2 [-1, 1]
img_f = img.astype(np.float32)
img_f = (img_f / 127.5) - 1.0
inp_tensor = np.expand_dims(img_f, axis=0)

# ── 3. INFERENCIA ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. INFERENCIA")
print("=" * 60)

interp.set_tensor(inp[0]['index'], inp_tensor)
interp.invoke()

mask_raw  = interp.get_tensor(outp[mask_idx]['index'])[0]   # (256, 256, 2)
class_raw = interp.get_tensor(outp[class_idx]['index'])[0]  # (2,)

print(f"\nClasificación:")
for i, v in enumerate(class_raw):
    print(f"  {CLASS_NAMES[i]}: {v*100:.2f}%")
pred_class = int(np.argmax(class_raw))
print(f"  → Predicción: {CLASS_NAMES[pred_class]}  ({class_raw[pred_class]*100:.1f}%)")

print(f"\nMáscara shape: {mask_raw.shape}")

# ── 4. DIAGNÓSTICO CANAL A CANAL ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. DISTRIBUCIÓN DE VALORES POR CANAL")
print("=" * 60)

for c in range(NUM_CLASSES):
    canal = mask_raw[:, :, c]
    print(f"\n  Canal {c} — {CLASS_NAMES[c]}:")
    print(f"    min   = {canal.min():.4f}")
    print(f"    max   = {canal.max():.4f}")
    print(f"    mean  = {canal.mean():.4f}")
    print(f"    >0.35 = {(canal > 0.35).sum()} píxeles  ({(canal > 0.35).mean()*100:.1f}%)")
    print(f"    >0.50 = {(canal > 0.50).sum()} píxeles  ({(canal > 0.50).mean()*100:.1f}%)")
    print(f"    >0.70 = {(canal > 0.70).sum()} píxeles  ({(canal > 0.70).mean()*100:.1f}%)")
    print(f"    >0.90 = {(canal > 0.90).sum()} píxeles  ({(canal > 0.90).mean()*100:.1f}%)")

# Argmax de la máscara (qué clase gana cada píxel)
argmax_mask = np.argmax(mask_raw, axis=-1)  # (256, 256)
print(f"\n  Argmax de máscara:")
for c in range(NUM_CLASSES):
    n = (argmax_mask == c).sum()
    print(f"    Clase {c} ({CLASS_NAMES[c]}) gana {n} píxeles  ({n/SIZE/SIZE*100:.1f}%)")

# ── 5. DIAGNÓSTICO CRÍTICO: ¿son los valores softmax válidos? ─────────────────
print("\n" + "=" * 60)
print("5. VALIDACIÓN SOFTMAX")
print("=" * 60)

sumas = mask_raw.sum(axis=-1)  # cada píxel debe sumar ~1.0
print(f"  Suma por píxel — min={sumas.min():.4f}  max={sumas.max():.4f}  mean={sumas.mean():.4f}")
if abs(sumas.mean() - 1.0) < 0.01:
    print("  ✅ Softmax válido — los canales suman ~1.0")
else:
    print("  ⚠️  NO es softmax — los canales NO suman 1.0 → el umbral 0.5 no aplica")
    print("      Esto explica por qué se pinta toda la imagen")
    print("      Solución: usar percentil en lugar de umbral fijo")

# ── 6. UMBRAL RECOMENDADO ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. UMBRAL RECOMENDADO PARA ANDROID")
print("=" * 60)

if pred_class != 1:  # si detectó enfermedad
    canal_enf = mask_raw[:, :, pred_class]
    # Solo píxeles donde la clase enferma gana el argmax
    mascara_argmax = (argmax_mask == pred_class)
    valores_ganadores = canal_enf[mascara_argmax]

    if len(valores_ganadores) > 0:
        p50 = np.percentile(valores_ganadores, 50)
        p70 = np.percentile(valores_ganadores, 70)
        p90 = np.percentile(valores_ganadores, 90)
        print(f"  Valores donde BacterialSpot gana el argmax: {len(valores_ganadores)} píxeles")
        print(f"  Percentil 50 = {p50:.4f}  → pinta el 50% más activo")
        print(f"  Percentil 70 = {p70:.4f}  → pinta el 30% más activo")
        print(f"  Percentil 90 = {p90:.4f}  → pinta el 10% más activo ← recomendado")
        print(f"\n  ✅ USA ESTE UMBRAL EN ANDROID: {p70:.2f}f")
    else:
        print("  ⚠️  Argmax no selecciona ningún píxel como BacterialSpot")
        print("      El modelo predice Healthy en TODOS los píxeles")
        print("      Problema: el clasificador dice enfermo pero la máscara dice sano")
else:
    print("  Hoja predicha como Healthy — no se esperan manchas")

# ── 7. VISUALIZACIÓN ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. GENERANDO VISUALIZACIÓN")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"Diagnóstico — Predicción: {CLASS_NAMES[pred_class]} ({class_raw[pred_class]*100:.1f}%)", fontsize=14)

axes[0, 0].imshow(img)
axes[0, 0].set_title("Imagen original")
axes[0, 0].axis('off')

axes[0, 1].imshow(mask_raw[:, :, 0], cmap='hot', vmin=0, vmax=1)
axes[0, 1].set_title(f"Canal 0 — {CLASS_NAMES[0]}\nmax={mask_raw[:,:,0].max():.3f}")
axes[0, 1].axis('off')
plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])

axes[0, 2].imshow(mask_raw[:, :, 1], cmap='hot', vmin=0, vmax=1)
axes[0, 2].set_title(f"Canal 1 — {CLASS_NAMES[1]}\nmax={mask_raw[:,:,1].max():.3f}")
axes[0, 2].axis('off')
plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])

axes[1, 0].imshow(argmax_mask, cmap='RdYlGn', vmin=0, vmax=1)
axes[1, 0].set_title("Argmax (0=Bacterial, 1=Healthy)")
axes[1, 0].axis('off')

# Overlay con umbral 0.50
overlay_50 = img.copy()
if pred_class == 0:
    mask_50 = (argmax_mask == 0) & (mask_raw[:, :, 0] > 0.50)
    overlay_50[mask_50] = [255, 140, 0]
axes[1, 1].imshow(overlay_50)
axes[1, 1].set_title(f"Overlay umbral 0.50\n({mask_50.sum() if pred_class==0 else 0} px)")
axes[1, 1].axis('off')

# Histograma de valores del canal enfermo
if pred_class == 0:
    axes[1, 2].hist(mask_raw[:, :, 0].flatten(), bins=50, color='orange', edgecolor='black')
    axes[1, 2].axvline(0.50, color='red',   linestyle='--', label='0.50')
    axes[1, 2].axvline(0.35, color='blue',  linestyle='--', label='0.35')
    axes[1, 2].set_title(f"Histograma Canal BacterialSpot")
    axes[1, 2].set_xlabel("Valor del píxel")
    axes[1, 2].set_ylabel("Frecuencia")
    axes[1, 2].legend()
else:
    axes[1, 2].text(0.5, 0.5, "Predicción: Healthy\nNo hay canal enfermo",
                    ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig("diagnostico_mascara.png", dpi=120, bbox_inches='tight')
plt.show()
print("\n✅ Guardado: diagnostico_mascara.png")
print("\n" + "=" * 60)
print("RESUMEN — pega estos valores aquí para continuar el diagnóstico")
print("=" * 60)
print(f"  Clasificación: {CLASS_NAMES[pred_class]} ({class_raw[pred_class]*100:.1f}%)")
print(f"  Canal 0 max  : {mask_raw[:,:,0].max():.4f}")
print(f"  Canal 1 max  : {mask_raw[:,:,1].max():.4f}")
print(f"  Softmax suma : {sumas.mean():.4f}")
print(f"  Argmax px cl0: {(argmax_mask==0).sum()}")
print(f"  Argmax px cl1: {(argmax_mask==1).sum()}")