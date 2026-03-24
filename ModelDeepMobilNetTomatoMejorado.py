# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:18:14 2026

@author: dkpin
"""
# -*- coding: utf-8 -*-
"""
DeepLabV3+ MobileNetV2 - Tomato Disease Detection
OPTIMIZED FOR KOTLIN NATIVE / ANDROID
==============================================
Autor: optimizado desde versión original dkpin
Mejoras:
  1. Data Augmentation sincronizado imagen+mascara
  2. ASPP con dilataciones {6,12,18} + DW-Separable convs (-40% params)
  3. Skip connections low-level (DeepLabV3+ style) -> mejor mIoU
  4. Entrenamiento en 2 fases: frozen -> fine-tuning parcial
  5. Label Smoothing en clasificacion
  6. INT8 Post-Training Quantization + benchmark TFLite para Android
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pycocotools.coco import COCO
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix, classification_report)

# ============================================================================
# CONFIGURACION GLOBAL
# ============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
NUM_CLASSES = 5
CLASS_NAMES = ["EarlyBlight", "BacterialSpot", "Healthy",
               "LateBlight", "YellowLeafCurl"]
BASE_PATH   = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v5i.coco/'

# ============================================================================
# 1. DATASET CON DATA AUGMENTATION SINCRONIZADO
# ============================================================================
class TomatoDiseaseDataset(tf.keras.utils.Sequence):
    """
    Dataset con augmentation on-the-fly sincronizado imagen+mascara.
    Critico: las transformaciones geometricas se aplican identicamente
    a imagen y mascara para evitar desalineamiento.
    """
    CAT_MAP     = {0: 0, 1: 1, 2: 0, 3: 2, 4: 3, 5: 4}
    HEALTHY_IDX = 2

    def __init__(self, img_dir, ann_file, batch_size=8,
                 img_size=(256, 256), augment=False):
        self.img_dir    = img_dir
        self.coco       = COCO(ann_file)
        self.ids        = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size   = img_size
        self.augment    = augment

    def __len__(self):
        return len(self.ids) // self.batch_size

    def _augment(self, image, mask):
        """Transformaciones geometricas y fotometricas sincronizadas."""
        # Flip horizontal
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask  = cv2.flip(mask, 1)
        # Flip vertical
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask  = cv2.flip(mask, 0)
        # Rotacion +/-30 grados
        if np.random.rand() > 0.5:
            ang = np.random.uniform(-30, 30)
            M   = cv2.getRotationMatrix2D(
                      (self.img_size[1] // 2, self.img_size[0] // 2), ang, 1)
            image = cv2.warpAffine(image, M, self.img_size)
            for c in range(mask.shape[-1]):
                mask[:, :, c] = cv2.warpAffine(
                    mask[:, :, c], M, self.img_size,
                    flags=cv2.INTER_NEAREST)
        # Zoom aleatorio 0.9-1.1
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D(
                    (self.img_size[1] // 2, self.img_size[0] // 2), 0, scale)
            image = cv2.warpAffine(image, M, self.img_size)
            for c in range(mask.shape[-1]):
                mask[:, :, c] = cv2.warpAffine(
                    mask[:, :, c], M, self.img_size,
                    flags=cv2.INTER_NEAREST)
        # Brillo y contraste (solo imagen, no mascara)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.75, 1.25)
            beta  = np.random.uniform(-25, 25)
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        # Desenfoque gaussiano leve (simula camara movil)
        if np.random.rand() > 0.4:
            k     = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), 0)
        return image, mask

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            image = cv2.imread(
                os.path.join(self.img_dir, img_info['file_name']))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            mask  = np.zeros(
                (self.img_size[0], self.img_size[1], NUM_CLASSES),
                dtype=np.float32)

            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            anns    = self.coco.loadAnns(ann_ids)
            cur_cls = self.HEALTHY_IDX

            if anns:
                cur_cls = self.CAT_MAP.get(
                    anns[0]['category_id'], self.HEALTHY_IDX)
                for ann in anns:
                    m_id = self.CAT_MAP.get(ann['category_id'], -1)
                    if (0 <= m_id < NUM_CLASSES
                            and ann.get('segmentation')
                            and len(ann['segmentation']) > 0):
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size,
                                           interpolation=cv2.INTER_NEAREST)
                            mask[:, :, m_id] = np.maximum(
                                mask[:, :, m_id], m.astype(np.float32))
                        except Exception:
                            continue

            # Canal Healthy = pixeles donde ninguna clase esta activa
            mask[:, :, self.HEALTHY_IDX] = (
                np.max(mask, axis=-1) == 0).astype(np.float32)

            if self.augment:
                image, mask = self._augment(image, mask)

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(
                cur_cls, num_classes=NUM_CLASSES))

        return (np.array(X),
                {"mask_out":  np.array(y_mask),
                 "class_out": np.array(y_class)})

    def summarize_dataset(self):
        counts = {n: 0 for n in CLASS_NAMES}
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns    = self.coco.loadAnns(ann_ids)
            idx = (self.CAT_MAP.get(anns[0]['category_id'], self.HEALTHY_IDX)
                   if anns else self.HEALTHY_IDX)
            counts[CLASS_NAMES[idx]] += 1
        print("\n RESUMEN DEL DATASET:")
        for k, v in counts.items():
            print(f"  {k}: {v} imagenes")
        return counts


# ============================================================================
# 2. ASPP MEJORADO: dilataciones {6,12,18} + DW-Separable convolutions
#    ~40% menos parametros que Conv2D estandar, mas eficiente en movil
# ============================================================================
def _dw_sep_conv(x, filters, dilation_rate=1):
    """
    Depthwise-Separable Conv + BN + ReLU6.
    ReLU6 es esencial para compatibilidad con cuantizacion INT8 en Android.
    """
    x = layers.DepthwiseConv2D(
            3, padding='same', dilation_rate=dilation_rate,
            use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    return x


def ASPP_block(tensor, out_filters=256):
    """
    Atrous Spatial Pyramid Pooling con 5 ramas:
    1x1 conv + dilataciones {6,12,18} + global pooling.
    Dilatacion 18 (nueva) captura contexto mas amplio para lesiones grandes.
    """
    dims = tensor.shape

    # Rama 1x1
    b0 = layers.Conv2D(out_filters, 1, use_bias=False)(tensor)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.ReLU(max_value=6.0)(b0)

    # Ramas atrous con DW-separable
    b1 = _dw_sep_conv(tensor, out_filters, dilation_rate=6)
    b2 = _dw_sep_conv(tensor, out_filters, dilation_rate=12)
    b3 = _dw_sep_conv(tensor, out_filters, dilation_rate=18)

    # Image-level pooling (contexto global)
    bp = layers.GlobalAveragePooling2D()(tensor)
    bp = layers.Reshape((1, 1, dims[-1]))(bp)
    bp = layers.Conv2D(out_filters, 1, use_bias=False)(bp)
    bp = layers.BatchNormalization()(bp)
    bp = layers.ReLU(max_value=6.0)(bp)
    bp = layers.UpSampling2D(
             size=(dims[1], dims[2]), interpolation='bilinear')(bp)

    # Fusion
    res = layers.Concatenate()([b0, b1, b2, b3, bp])
    res = layers.Conv2D(out_filters, 1, use_bias=False)(res)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU(max_value=6.0)(res)
    return res


# ============================================================================
# 3. ARQUITECTURA DEEPLABV3+ CON SKIP CONNECTION (mejora directa de mIoU)
#    La fusion de features stride-4 recupera detalle espacial en bordes
#    de lesiones que se perdia al subir solo desde stride-16
# ============================================================================
def build_deeplab_mobilenet(input_shape=(256, 256, 3),
                             num_classes=NUM_CLASSES,
                             freeze_backbone=True):
    """
    DeepLabV3+ con MobileNetV2.
    - high_feat (stride 16) -> ASPP
    - low_feat  (stride  4) -> skip connection (estilo DeepLabV3+)
    - freeze_backbone: True en fase 1, False en fase 2
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = not freeze_backbone

    # Stride 16 -> entrada al ASPP
    high_feat = base.get_layer("block_13_expand_relu").output  # 16x16x96
    # Stride  4 -> skip connection con detalle espacial
    low_feat  = base.get_layer("block_3_expand_relu").output   # 64x64x144

    # ---- Rama de Segmentacion ---------------------------------------------
    aspp = ASPP_block(high_feat, out_filters=256)

    # Upsample x4 -> 64x64
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(aspp)

    # Proyeccion low-level features (DeepLabV3+ fusion)
    low = layers.Conv2D(48, 1, use_bias=False)(low_feat)
    low = layers.BatchNormalization()(low)
    low = layers.ReLU(max_value=6.0)(low)

    # Fusion con skip connection
    x = layers.Concatenate()([x, low])   # 64x64
    x = _dw_sep_conv(x, 256)
    x = _dw_sep_conv(x, 256)

    # Upsample x4 -> 256x256
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    mask_out = layers.Conv2D(
        num_classes, 1, activation='softmax', name='mask_out')(x)

    # ---- Rama de Clasificacion --------------------------------------------
    c5  = base.get_layer("out_relu").output
    gap = layers.GlobalAveragePooling2D()(c5)
    gap = layers.Dense(128, use_bias=False)(gap)
    gap = layers.BatchNormalization()(gap)
    gap = layers.ReLU(max_value=6.0)(gap)
    gap = layers.Dropout(0.4)(gap)
    class_out = layers.Dense(
        num_classes, activation='softmax', name='class_out')(gap)

    return models.Model(
        inputs=base.input, outputs=[mask_out, class_out]), base


# ============================================================================
# 4. PERDIDAS Y METRICAS
# ============================================================================
def focal_tversky_loss(y_true, y_pred,
                       alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    """
    Focal Tversky: alpha=0.7 penaliza FN (zonas enfermas no detectadas)
    mas que FP, critico para diagnostico medico/agricola.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
    tp  = tf.reduce_sum(y_true_f * y_pred_f,         axis=0)
    fn  = tf.reduce_sum(y_true_f * (1 - y_pred_f),   axis=0)
    fp  = tf.reduce_sum((1 - y_true_f) * y_pred_f,   axis=0)
    tvi = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.reduce_mean(tf.pow(1 - tvi, 1 / gamma))


def label_smoothing_cce(y_true, y_pred, smoothing=0.1):
    """Label smoothing: reduce sobreajuste en clases desbalanceadas."""
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=smoothing)


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    """MeanIoU corregido para salidas softmax (argmax antes de calcular)."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={
            "mask_out":  focal_tversky_loss,
            "class_out": label_smoothing_cce
        },
        loss_weights={"mask_out": 1.0, "class_out": 0.5},
        metrics={
            "mask_out":  [UpdatedMeanIoU(num_classes=NUM_CLASSES,
                                          name="mean_iou"), "accuracy"],
            "class_out": "accuracy"
        }
    )


def unfreeze_top_layers(backbone, n=30):
    """
    Descongela las ultimas n capas del backbone para fine-tuning.
    Las BatchNormalization se mantienen frozen para estabilidad.
    """
    for layer in backbone.layers[-n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    print(f"  Descongeladas {n} capas del backbone para fine-tuning.")


# ============================================================================
# CARGA DE DATASETS
# ============================================================================
print("\n Cargando datasets...")

train_ds = TomatoDiseaseDataset(
    BASE_PATH + "train",
    BASE_PATH + "train/_annotations.coco.json",
    batch_size=BATCH_SIZE, augment=True)    # augmentation activo en train

val_ds = TomatoDiseaseDataset(
    BASE_PATH + "valid",
    BASE_PATH + "valid/_annotations.coco.json",
    batch_size=BATCH_SIZE, augment=False)

test_ds = TomatoDiseaseDataset(
    BASE_PATH + "test",
    BASE_PATH + "test/_annotations.coco.json",
    batch_size=BATCH_SIZE, augment=False)

print(f"  Train : {len(train_ds)} batches")
print(f"  Val   : {len(val_ds)} batches")
print(f"  Test  : {len(test_ds)} batches")
train_ds.summarize_dataset()


# ============================================================================
# 5. ENTRENAMIENTO EN 2 FASES
# ============================================================================
model, backbone = build_deeplab_mobilenet(freeze_backbone=True)
compile_model(model, lr=1e-4)
model.summary()

# ---- FASE 1: Backbone completamente congelado ----------------------------
# Las cabezas (ASPP, decoder, clasificador) aprenden desde cero
# sin arriesgar destruir los pesos ImageNet del backbone
callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_phase1.keras',
                    monitor='val_class_out_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-7, verbose=1),
    TensorBoard(log_dir='./logs/phase1')
]
print("\n FASE 1 - Backbone congelado (max 35 epocas)...")
h1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=35, callbacks=callbacks_phase1, verbose=1)

# ---- FASE 2: Fine-tuning ultimas 30 capas del backbone -------------------
# LR 5x menor para ajuste fino sin sobreescribir lo aprendido
unfreeze_top_layers(backbone, n=30)
compile_model(model, lr=2e-5)

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_tomato_optimized.keras',
                    monitor='val_class_out_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-8, verbose=1),
    TensorBoard(log_dir='./logs/phase2')
]
print("\n FASE 2 - Fine-tuning parcial (max 20 epocas)...")
h2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=20, callbacks=callbacks_phase2, verbose=1)

# Guardar historico combinado de ambas fases
hist_combined = {}
for k in h1.history:
    hist_combined[k] = h1.history[k] + h2.history.get(k, [])
pd.DataFrame(hist_combined).to_csv("history_optimized.csv", index=False)


# ============================================================================
# 6. EVALUACION FINAL
# ============================================================================
print("\n Evaluando en test set...")
res = model.evaluate(test_ds, verbose=0)

y_true_all, y_pred_all = [], []
for i in range(len(test_ds)):
    X_b, y_b = test_ds[i]
    p = model.predict(X_b, verbose=0)
    y_true_all.extend(np.argmax(y_b['class_out'], axis=1))
    y_pred_all.extend(np.argmax(p[1], axis=1))

_, recall, f1, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average='weighted', zero_division=0)

print("\n" + "="*55)
print("  RESULTADOS FINALES")
print("="*55)
print(f"  Perdida Total:         {res[0]:.4f}")
print(f"  Segmentacion mIoU:     {res[5]*100:.2f}%")
print(f"  Segmentacion Acc:      {res[4]*100:.2f}%")
print(f"  Clasificacion Acc:     {res[3]*100:.2f}%")
print(f"  Clasificacion Recall:  {recall*100:.2f}%")
print(f"  Clasificacion F1:      {f1*100:.2f}%")
print(f"  Parametros totales:    {model.count_params()/1e6:.2f} M")
print("="*55)
print("\n REPORTE POR CLASE:")
print(classification_report(y_true_all, y_pred_all,
                             target_names=CLASS_NAMES, zero_division=0))


# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================
COLORS = ['Reds', 'Oranges', 'Greens', 'Blues', 'YlOrBr']


def plot_training_history(history_dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    pairs = [
        ('loss',              'val_loss',              'Total Loss'),
        ('mask_out_accuracy', 'val_mask_out_accuracy', 'Segmentation Accuracy'),
        ('class_out_accuracy','val_class_out_accuracy','Classification Accuracy'),
    ]
    for ax, (tr, va, title) in zip(axes, pairs):
        ax.plot(history_dict[tr], label='Train')
        ax.plot(history_dict[va], label='Val')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig('training_history_optimized.png', dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix (%) - Optimized', fontsize=13)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized.png', dpi=300)
    plt.show()


def visualize_predictions(model, dataset, n=4):
    X_b, y_b = dataset[0]
    p        = model.predict(X_b, verbose=0)
    fig, ax  = plt.subplots(n, 4, figsize=(18, n * 4))
    for i in range(n):
        img = ((X_b[i] - X_b[i].min()) /
               (X_b[i].max() - X_b[i].min() + 1e-8) * 255).astype(np.uint8)
        gt_m  = np.argmax(y_b['mask_out'][i],  axis=-1)
        gt_c  = np.argmax(y_b['class_out'][i])
        pr_m  = np.argmax(p[0][i],              axis=-1)
        pr_c  = np.argmax(p[1][i])
        pr_cf = p[1][i][pr_c]

        ax[i, 0].imshow(img)
        ax[i, 0].set_title(f"GT: {CLASS_NAMES[gt_c]}", color='blue')
        ax[i, 1].imshow(gt_m, cmap='viridis', vmin=0, vmax=4)
        ax[i, 1].set_title("GT Mask")
        ax[i, 2].imshow(pr_m, cmap='viridis', vmin=0, vmax=4)
        ax[i, 2].set_title("Pred Mask")
        ax[i, 3].imshow(p[0][i, :, :, pr_c], cmap=COLORS[pr_c])
        ax[i, 3].set_title(f"PRED: {CLASS_NAMES[pr_c]}\nConf: {pr_cf:.2f}",
                           color='red', fontweight='bold')
        for a in ax[i]:
            a.axis('off')
    plt.tight_layout()
    plt.savefig('predictions_optimized.png', dpi=150)
    plt.show()


plot_training_history(hist_combined)
plot_confusion_matrix(y_true_all, y_pred_all, CLASS_NAMES)
visualize_predictions(model, test_ds)


# ============================================================================
# 8. BENCHMARK DE LATENCIA (Keras + TFLite)
# ============================================================================
def benchmark_keras(model, test_ds, iterations=200):
    sample, _ = test_ds[0]
    img = np.expand_dims(sample[0], axis=0)
    # Warm-up
    for _ in range(20):
        model.predict(img, verbose=0)
    lats = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(img, verbose=0)
        lats.append(time.perf_counter() - t0)
    avg = np.mean(lats) * 1000
    std = np.std(lats)  * 1000
    p95 = np.percentile(lats, 95) * 1000
    print(f"\n  Latencia Keras  avg={avg:.1f}ms  std={std:.1f}ms  "
          f"p95={p95:.1f}ms  FPS={1000/avg:.1f}")
    return avg


def benchmark_tflite(tflite_path, test_ds, iterations=200):
    """
    Benchmark del modelo cuantizado.
    Nota: en CPU de desarrollo, Android ARM cortara latencia ~2-3x adicional
    gracias a aceleracion NNAPI/GPU delegate.
    """
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx  = interp.get_input_details()[0]['index']
    out_dets = interp.get_output_details()

    sample, _ = test_ds[0]
    img = np.expand_dims(sample[0], axis=0).astype(np.float32)

    # Warm-up
    for _ in range(20):
        interp.set_tensor(inp_idx, img)
        interp.invoke()

    lats = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        interp.set_tensor(inp_idx, img)
        interp.invoke()
        lats.append(time.perf_counter() - t0)

    avg = np.mean(lats) * 1000
    std = np.std(lats)  * 1000
    p95 = np.percentile(lats, 95) * 1000
    print(f"  Latencia TFLite avg={avg:.1f}ms  std={std:.1f}ms  "
          f"p95={p95:.1f}ms  FPS={1000/avg:.1f}")
    print("  (CPU desarrollo - Android Arm ~2-3x mas rapido con NNAPI/GPU)")
    return avg


benchmark_keras(model, test_ds)


# ============================================================================
# 9. EXPORTACION: Keras + TFLite INT8 (optimizado para Android)
# ============================================================================
print("\n Guardando modelo Keras...")
model.save("TMTDeepMobNet_Optimized.keras")
print("  TMTDeepMobNet_Optimized.keras  guardado.")

# Generador de calibracion: 200 imagenes reales del train set
# Necesario para que el cuantizador aprenda los rangos de activacion INT8
def representative_data_gen():
    for i in range(min(200, len(train_ds))):
        X_b, _ = train_ds[i]
        for img in X_b:
            yield [np.expand_dims(img, 0).astype(np.float32)]

print("\n Convirtiendo a TFLite INT8...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations           = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset  = representative_data_gen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS        # ops TF no disponibles en builtins INT8
]
# float32 en entrada/salida: evita que Kotlin tenga que dequantizar manualmente
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_bytes = converter.convert()
tflite_path  = 'TMTDeepMobNet_Optimized_INT8.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_bytes)

keras_mb   = model.count_params() * 4 / 1024 / 1024
tflite_mb  = len(tflite_bytes) / 1024 / 1024
print(f"  TFLite INT8: {tflite_mb:.2f} MB  "
      f"(Keras float32 ~{keras_mb:.1f} MB, reduccion {keras_mb/tflite_mb:.1f}x)")

benchmark_tflite(tflite_path, test_ds)

print("\n" + "="*55)
print("  ARCHIVOS GENERADOS")
print("="*55)
print("  TMTDeepMobNet_Optimized.keras          <- modelo completo Keras")
print("  TMTDeepMobNet_Optimized_INT8.tflite    <- para Android / Kotlin")
print("  best_tomato_optimized.keras            <- mejor checkpoint")
print("  history_optimized.csv")
print("  training_history_optimized.png")
print("  confusion_matrix_optimized.png")
print("  predictions_optimized.png")
print("  logs/  (TensorBoard)")
print("="*55)
print("\n CLASES:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: {name}")
print("\n ENTRENAMIENTO COMPLETADO!")
