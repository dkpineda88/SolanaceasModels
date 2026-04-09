# -*- coding: utf-8 -*-
"""
Potato Disease Segmentation & Classification
Arquitectura: U-Net + MobileNetV2 (encoder preentrenado ImageNet)
Clases: Background/Other, EarlyBlight, Healthy, LateBlight

Basado en el pipeline original (DeepLab+MobileNetV2),
adaptado a estructura U-Net con skip connections en 5 niveles.
"""

import os
import time
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================
IMG_SIZE       = (256, 256)
BATCH_SIZE     = 8
NUM_CLASSES_SEG = 4   # Background/Other, EarlyBlight, Healthy, LateBlight
NUM_CLASSES_CLF = 4
CLASS_NAMES    = ["Background/Other", "EarlyBlight", "Healthy", "LateBlight"]

base_path = "D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v6i.coco/"  # Ajusta tu ruta

# ============================================================================
# 2. DATASET  (idéntico al original, sin cambios)
# ============================================================================
class PotatoDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256), augment=False):
        self.img_dir    = img_dir
        self.coco       = COCO(ann_file)
        self.ids        = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size   = img_size
        self.augment    = augment
        self.cat_map    = {0: 0, 1: 1, 2: 2, 3: 3}

    def __len__(self):
        return len(self.ids) // self.batch_size

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            image    = cv2.imread(os.path.join(self.img_dir, img_info['file_name']))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)

            mask_single = np.zeros(self.img_size, dtype=np.uint8)
            anns        = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

            clf_idx = 2  # Healthy por defecto (índice 2)
            if anns:
                categories = [self.cat_map.get(a['category_id'], 2) for a in anns]
                clf_idx    = max(categories) if categories else 2

                for ann in anns:
                    if not ann.get('segmentation'):
                        continue
                    m_id = self.cat_map.get(ann['category_id'], 0)
                    try:
                        m = self.coco.annToMask(ann)
                        m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                        mask_single[m.astype(bool)] = m_id
                    except:
                        continue

            # Aumento de datos simple
            if self.augment:
                if np.random.rand() > 0.5:          # Flip horizontal
                    image       = cv2.flip(image, 1)
                    mask_single = cv2.flip(mask_single, 1)
                if np.random.rand() > 0.5:          # Flip vertical
                    image       = cv2.flip(image, 0)
                    mask_single = cv2.flip(mask_single, 0)

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(tf.keras.utils.to_categorical(mask_single, num_classes=NUM_CLASSES_SEG))
            y_class.append(tf.keras.utils.to_categorical(clf_idx,    num_classes=NUM_CLASSES_CLF))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

    def on_epoch_end(self):
        np.random.shuffle(self.ids)


# ============================================================================
# 3. BLOQUE DECODIFICADOR U-NET
# ============================================================================
def unet_decoder_block(x, skip, filters):
    """
    Bloque estándar U-Net:
      1. UpSampling2D x2
      2. Concatenate con skip connection
      3. Conv2D + BN + ReLU  (x2)
    """
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    # Ajustar spatial dims si hay discrepancia (puede pasar con MobileNetV2)
    if skip is not None:
        # Recortar o padear para que coincidan
        x    = layers.Resizing(skip.shape[1], skip.shape[2])(x)
        x    = layers.Concatenate()([x, skip])

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


# ============================================================================
# 4. MODELO  U-Net + MobileNetV2
# ============================================================================
def build_unet_mobilenet(input_shape=(256, 256, 3)):
    """
    Encoder : MobileNetV2 preentrenado (ImageNet), congelado parcialmente.
    Skip connections en 5 escalas (igual que U-Net estándar):

      Escala  | Capa MobileNetV2             | Resolución (para 256x256)
      --------|------------------------------|---------------------------
      s1      | input_1  (imagen original)   | 256×256
      s2      | block_1_expand_relu           |  128×128
      s3      | block_3_expand_relu           |   64×64
      s4      | block_6_expand_relu           |   32×32
      s5      | block_13_expand_relu          |   16×16
      bottle  | out_relu  (salida encoder)   |    8×8

    Decoder : 5 bloques unet_decoder_block (256→128→64→32→16→8 canales)
    Salidas : mask_out (softmax 4 clases)  +  class_out (softmax 4 clases)
    """
    inputs     = tf.keras.Input(shape=input_shape)
    base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")

    # ---------- Skip connections ----------
    s1 = inputs                                                    # 256×256×3
    s2 = base_model.get_layer("block_1_expand_relu").output        # 128×128×96
    s3 = base_model.get_layer("block_3_expand_relu").output        #  64×64×144
    s4 = base_model.get_layer("block_6_expand_relu").output        #  32×32×192
    s5 = base_model.get_layer("block_13_expand_relu").output       #  16×16×576

    # ---------- Bottleneck ----------
    bottleneck = base_model.get_layer("out_relu").output           #   8×8×1280

    # ---------- Rama de Clasificación (desde bottleneck) ----------
    gap       = layers.GlobalAveragePooling2D()(bottleneck)
    gap       = layers.Dropout(0.3)(gap)
    gap       = layers.Dense(128, activation="relu")(gap)
    gap       = layers.Dropout(0.2)(gap)
    class_out = layers.Dense(NUM_CLASSES_CLF, activation="softmax", name="class_out")(gap)

    # ---------- Decoder U-Net ----------
    d5 = unet_decoder_block(bottleneck, s5, filters=256)   #  16×16×256
    d4 = unet_decoder_block(d5,         s4, filters=128)   #  32×32×128
    d3 = unet_decoder_block(d4,         s3, filters=64)    #  64×64×64
    d2 = unet_decoder_block(d3,         s2, filters=32)    # 128×128×32
    d1 = unet_decoder_block(d2,         s1, filters=16)    # 256×256×16

    # ---------- Cabeza de Segmentación ----------
    mask_out = layers.Conv2D(NUM_CLASSES_SEG, 1, activation="softmax", name="mask_out")(d1)

    model = models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

    # Congelar las primeras 100 capas del encoder (fine-tuning parcial)
    for layer in base_model.layers[:100]:
        layer.trainable = False

    return model


# ============================================================================
# 5. MÉTRICAS Y PÉRDIDAS PERSONALIZADAS
# ============================================================================
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=1.33):
    """Focal Tversky Loss: penaliza más los falsos negativos (útil con clases desbalanceadas)."""
    def loss(y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES_SEG]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES_SEG]), tf.float32)
        tp  = tf.reduce_sum(y_true_f * y_pred_f,           axis=0)
        fn  = tf.reduce_sum(y_true_f * (1 - y_pred_f),    axis=0)
        fp  = tf.reduce_sum((1 - y_true_f) * y_pred_f,    axis=0)
        tvi = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
        return tf.reduce_mean(tf.pow(1.0 - tvi, gamma))
    return loss


# ============================================================================
# 6. COMPILACIÓN Y ENTRENAMIENTO
# ============================================================================
model = build_unet_mobilenet()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "mask_out":  focal_tversky_loss(),
        "class_out": "categorical_crossentropy"
    },
    loss_weights={"mask_out": 1.5, "class_out": 0.5},
    metrics={
        "mask_out":  [UpdatedMeanIoU(num_classes=NUM_CLASSES_SEG, name="miou"), "accuracy"],
        "class_out": "accuracy"
    }
)

# Datasets
train_ds = PotatoDataset(base_path + "train", base_path + "train/_annotations.coco.json",
                         batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=True)
val_ds   = PotatoDataset(base_path + "valid", base_path + "valid/_annotations.coco.json",
                         batch_size=BATCH_SIZE, img_size=IMG_SIZE)
test_ds  = PotatoDataset(base_path + "test",  base_path + "test/_annotations.coco.json",
                         batch_size=BATCH_SIZE, img_size=IMG_SIZE)

callbacks = [
    ModelCheckpoint("best_potato_unet_mobilenet.keras", save_best_only=True, monitor="val_loss"),
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

print("🚀 Iniciando entrenamiento: U-Net + MobileNetV2 (EarlyBlight, Healthy, LateBlight)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

# ============================================================================
# 7. EVALUACIÓN
# ============================================================================
print("\n📦 Evaluando en test set...")
results = model.evaluate(test_ds, verbose=0)

y_true_list, y_pred_list = [], []
for i in range(len(test_ds)):
    X, y    = test_ds[i]
    preds   = model.predict(X, verbose=0)
    y_true_list.extend(np.argmax(y["class_out"],  axis=1))
    y_pred_list.extend(np.argmax(preds[1],         axis=1))

_, recall, f1, _ = precision_recall_fscore_support(y_true_list, y_pred_list, average="weighted")

print("\n" + "=" * 50)
print("🎯 RESULTADOS FINALES  —  U-Net + MobileNetV2")
print("=" * 50)
print(f"📉 Pérdida Total:          {results[0]:.4f}")
print(f"🎨 Segmentación  IoU:      {results[5] * 100:.2f}%")
print(f"🎨 Segmentación  Acc:      {results[4] * 100:.2f}%")
print(f"🏷️  Clasificación Acc:     {results[3] * 100:.2f}%")
print(f"🏷️  Clasificación Recall:  {recall * 100:.2f}%")
print(f"🏷️  Clasificación F1:      {f1 * 100:.2f}%")
print("=" * 50)

# Guardar historial
pd.DataFrame(history.history).to_csv("history_unet_mobilenet_potato8.csv", index=False)

# ============================================================================
# 8. PARÁMETROS Y LATENCIA
# ============================================================================
total_params = model.count_params() / 1e6
sample_img   = np.expand_dims(np.zeros((256, 256, 3)), 0)

# Warm-up
for _ in range(10):
    model.predict(sample_img, verbose=0)

start = time.time()
for _ in range(100):
    model.predict(sample_img, verbose=0)
avg_ms = ((time.time() - start) / 100) * 1000

print(f"\n📐 Parámetros totales: {total_params:.2f}M")
print(f"⏱️  Latencia promedio:  {avg_ms:.2f} ms  ({1000/avg_ms:.1f} FPS)")

# ============================================================================
# 9. EXPORTACIÓN TFLite
# ============================================================================
print("\n═══ Exportando a TFLite ═══")
model.save("PotatoUNetMobilenet8.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
converter.target_spec.supported_ops   = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()
tflite_path  = "PotatoUNetMobilenet8.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite guardado: {tflite_path}")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")


import time
import numpy as np

def calculate_latency(model, test_ds, iterations=100):
    # 1. Warm-up (Calentamiento)
    # Hacemos unas predicciones iniciales para "despertar" la GPU
    print("🔥 Calentando el modelo...")
    sample_img, _ = test_ds[0]
    single_img = np.expand_dims(sample_img[0], axis=0)
    for _ in range(10):
        _ = model.predict(single_img, verbose=0)

    # 2. Medición de Latencia
    print(f"⏱️ Midiendo latencia sobre {iterations} iteraciones...")
    latencies = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        
        # Ejecutar inferencia
        _ = model.predict(single_img, verbose=0)
        
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    # 3. Resultados
    avg_latency = np.mean(latencies) * 1000  # Convertir a milisegundos
    std_latency = np.std(latencies) * 1000
    fps = 1000 / avg_latency

    print("\n" + "="*45)
    print("🚀 REPORTE DE LATENCIA")
    print("="*45)
    print(f"Latencia Promedio: {avg_latency:.2f} ms")
    print(f"Desviación Estándar: {std_latency:.2f} ms")
    print(f"FPS Estimados:      {fps:.1f} cuadros/seg")
    print("="*45)
    
    return avg_latency
latency = calculate_latency(model, test_ds)

# ============================================================================
# 10. MATRIZ DE CONFUSIÓN
# ============================================================================
CLASS_NAMES_SHORT = ["EarlyBlight", "Healthy", "LateBlight"]

# Filtrar predicciones con índice > 0 (excluir Background del reporte de clases)
y_true_filt = [v - 1 for v in y_true_list  if v > 0]
y_pred_filt = [p - 1 for p, v in zip(y_pred_list, y_true_list) if v > 0]

cm      = confusion_matrix(y_true_filt, y_pred_filt)
cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8, 6))
sns.heatmap(cm_perc, annot=True, fmt=".1f", cmap="Greens",
            xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT)
plt.title("Confusion Matrix (%) — Potato U-Net + MobileNetV2")
plt.tight_layout()
plt.savefig("final_cm_unet_mobilenet.png", dpi=150)
plt.show()

# ============================================================================
# 11. CURVAS DE ENTRENAMIENTO
# ============================================================================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Total Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history["mask_out_accuracy"],     label="Train Mask Acc")
plt.plot(history.history["val_mask_out_accuracy"], label="Val Mask Acc")
plt.title("Segmentation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history["class_out_accuracy"],     label="Train Class Acc")
plt.plot(history.history["val_class_out_accuracy"], label="Val Class Acc")
plt.title("Classification Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig("training_history_unet_mobilenet.png", dpi=150)
plt.show()

# ============================================================================
# 12. VISUALIZACIÓN DE PREDICCIONES (4 ejemplos)
# ============================================================================
print("\n🎨 Generando visualizaciones...")

X_batch, y_batch = test_ds[0]
predictions = model.predict(X_batch, verbose=0)
mask_preds  = predictions[0]
class_preds = predictions[1]

colors      = ["YlOrBr", "Reds", "Greens", "Blues"]
num_samples = 4
fig, axes   = plt.subplots(num_samples, 4, figsize=(18, num_samples * 4))

for i in range(num_samples):
    img_vis = X_batch[i]
    img_min, img_max = img_vis.min(), img_vis.max()
    img_vis = ((img_vis - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    gt_mask       = np.argmax(y_batch["mask_out"][i], axis=-1)
    gt_class_idx  = np.argmax(y_batch["class_out"][i])
    pred_mask     = np.argmax(mask_preds[i],  axis=-1)
    pred_class    = np.argmax(class_preds[i])
    pred_conf     = class_preds[i][pred_class]

    axes[i, 0].imshow(img_vis)
    axes[i, 0].set_title(f"REAL: {CLASS_NAMES[gt_class_idx]}", fontsize=10, color="blue", fontweight="bold")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(gt_mask, cmap="viridis", vmin=0, vmax=NUM_CLASSES_SEG - 1)
    axes[i, 1].set_title("Ground Truth Mask", fontsize=10)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_mask, cmap="viridis", vmin=0, vmax=NUM_CLASSES_SEG - 1)
    axes[i, 2].set_title("Predicted Mask", fontsize=10)
    axes[i, 2].axis("off")

    heatmap = mask_preds[i, :, :, pred_class]
    axes[i, 3].imshow(heatmap, cmap=colors[pred_class])
    color_title = "red" if pred_class == 0 else "green"
    axes[i, 3].set_title(f"PRED: {CLASS_NAMES[pred_class]}\nConf: {pred_conf:.2f}",
                          fontsize=10, fontweight="bold", color=color_title)
    axes[i, 3].axis("off")

plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.suptitle("Resultados — Potato U-Net + MobileNetV2", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("predictions_unet_mobilenet.png", dpi=150)
plt.show()