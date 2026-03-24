# -*- coding: utf-8 -*-
"""
MaskMobileNet CORREGIDO — Bell Pepper
Problema original: decoder demasiado superficial + backbone libre desde el inicio
→ el modelo aprende a clasificar pero la máscara dice "todo enfermo"

Fixes aplicados:
1. Decoder con pasos graduales (x2 x2 x2 x2) en lugar de x4+x4
2. Fase 1: backbone congelado → el decoder aprende a segmentar
3. Fase 2: fine-tuning solo de las últimas capas del backbone
4. Loss con peso mayor en segmentación para forzar aprendizaje de máscara
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import pandas as pd
import time

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
CLASS_NAMES = ["Bacterial Spot", "Healthy"]
NUM_CLASSES = len(CLASS_NAMES)
HEALTHY_IDX = 1

base_path = "D:/DATASETS/Imagenes/Solanaceas/BellPepper.v1i.coco/"

# ============================================================================
# 2. DATASET CON AUGMENTACIÓN REAL
# ============================================================================
class BellPepperDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8,
                 img_size=(256, 256), augment=False):
        self.img_dir     = img_dir
        self.coco        = COCO(ann_file)
        self.batch_size  = batch_size
        self.img_size    = img_size
        self.augment     = augment
        self.cat_map     = {0: 0, 1: 0, 2: 1}
        self.healthy_idx = HEALTHY_IDX

        all_ids = list(self.coco.imgs.keys())

        # Separar por clase para balanceo
        self.disease_ids = []
        self.healthy_ids = []
        for img_id in all_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns    = self.coco.loadAnns(ann_ids)
            mapped  = [self.cat_map.get(a['category_id'], 1) for a in anns]
            if anns and 0 in mapped:
                self.disease_ids.append(img_id)
            else:
                self.healthy_ids.append(img_id)

        print(f"  BacterialSpot: {len(self.disease_ids)} | Healthy: {len(self.healthy_ids)}")
        self.ids = self._balanced_sample()

        # Augmentación con Albumentations
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.05, p=0.6),
            A.GaussNoise(var_limit=(5, 25), p=0.3),
            A.ElasticTransform(alpha=1, sigma=5, p=0.2),
        ])

    def _balanced_sample(self):
        n_disease = len(self.disease_ids)
        n_healthy = len(self.healthy_ids)
        n = n_disease if n_disease < n_healthy else n_healthy  # sin min()
        
        d_arr = np.array(self.disease_ids, dtype=object)
        h_arr = np.array(self.healthy_ids, dtype=object)
        d = np.random.choice(d_arr, n, replace=False).tolist()
        h = np.random.choice(h_arr, n, replace=False).tolist()
        combined = d + h
        np.random.shuffle(combined)
        return combined

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
            ann_ids     = self.coco.getAnnIds(imgIds=img_id)
            anns        = self.coco.loadAnns(ann_ids)

            current_class = self.healthy_idx
            if anns:
                mapped = [self.cat_map.get(a['category_id'], 1) for a in anns]
                current_class = 0 if 0 in mapped else 1
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= m_id < NUM_CLASSES:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size,
                                           interpolation=cv2.INTER_NEAREST)
                            mask_single[m.astype(bool)] = m_id
                        except:
                            continue

            if mask_single.max() == 0 and not anns:
                mask_single[:] = self.healthy_idx

            # Augmentación conectada
            if self.augment:
                aug = self.transform(image=image, mask=mask_single)
                image, mask_single = aug['image'], aug['mask']

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(tf.keras.utils.to_categorical(
                mask_single, num_classes=NUM_CLASSES))
            y_class.append(tf.keras.utils.to_categorical(
                current_class, num_classes=NUM_CLASSES))

        if len(X) == 0:
            dummy_x = np.zeros((1, *self.img_size, 3))
            dummy_m = np.zeros((1, *self.img_size, NUM_CLASSES))
            dummy_c = np.zeros((1, NUM_CLASSES))
            return dummy_x, {"mask_out": dummy_m, "class_out": dummy_c}

        return (np.array(X),
                {"mask_out":  np.array(y_mask),
                 "class_out": np.array(y_class)})

    def on_epoch_end(self):
        self.ids = self._balanced_sample()


# ============================================================================
# 3. ARQUITECTURA CORREGIDA
#    Decoder gradual x2→x2→x2→x2 con skip connections en cada paso
#    (En lugar del x4+x4 original que causaba máscaras borrosas)
# ============================================================================
def build_mask_mobilenet_fixed(input_shape=(256, 256, 3)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False  # CRÍTICO: congelar en Fase 1

    # Skip connections en resoluciones progresivas
    s1 = base.get_layer("block_1_expand_relu").output   # 128x128 — 96ch
    s2 = base.get_layer("block_3_expand_relu").output   # 64x64  — 144ch
    s3 = base.get_layer("block_6_expand_relu").output   # 32x32  — 192ch
    s4 = base.get_layer("block_13_expand_relu").output  # 16x16  — 576ch
    bridge = base.get_layer("out_relu").output           # 8x8   — 1280ch

    def decoder_block(inputs, skip, filters):
        # Subida x2 + skip + refinamiento
        x = layers.UpSampling2D((2, 2), interpolation="bilinear")(inputs)
        # Ajuste dinámico de tamaño por si hay mismatch
        x = layers.Lambda(
            lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3],
                                       method="bilinear")
        )([x, skip])
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        return x

    # Decoder: 8→16→32→64→128→256
    d1 = decoder_block(bridge, s4, 256)   # 16x16
    d2 = decoder_block(d1,     s3, 128)   # 32x32
    d3 = decoder_block(d2,     s2,  64)   # 64x64
    d4 = decoder_block(d3,     s1,  32)   # 128x128

    # Último upsampling a 256x256 (sin skip disponible)
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(d4)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    # Salida de segmentación
    mask_out = layers.Conv2D(NUM_CLASSES, 1, activation="softmax",
                              name="mask_out")(x)

    # Cabeza de clasificación — usa múltiples niveles del encoder
    gap_bridge = layers.GlobalAveragePooling2D()(bridge)   # 1280
    gap_s4     = layers.GlobalAveragePooling2D()(s4)       # 576
    gap_s3     = layers.GlobalAveragePooling2D()(s3)       # 192
    combined   = layers.Concatenate()([gap_bridge, gap_s4, gap_s3])
    fc  = layers.Dense(256, activation="relu")(combined)
    fc  = layers.BatchNormalization()(fc)
    fc  = layers.Dropout(0.5)(fc)
    fc  = layers.Dense(64, activation="relu")(fc)
    fc  = layers.Dropout(0.3)(fc)
    class_out = layers.Dense(NUM_CLASSES, activation="softmax",
                              name="class_out")(fc)

    return models.Model(inputs=base.input,
                        outputs=[mask_out, class_out])


# ============================================================================
# 4. MÉTRICAS Y PÉRDIDA
# ============================================================================
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=1.33):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
        tp  = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        fn  = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
        fp  = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
        tvi = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
        return tf.reduce_mean(tf.pow(1.0 - tvi, gamma))
    return loss


def combined_seg_loss(alpha=0.7, beta=0.3, gamma=1.33,
                      w_tversky=0.6, w_ce=0.4):
    ftl = focal_tversky_loss(alpha, beta, gamma)
    def loss(y_true, y_pred):
        return (w_tversky * ftl(y_true, y_pred)
                + w_ce * tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_true, y_pred)))
    return loss


def compile_model(m, lr):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
            # Peso 1.5 en segmentación para forzar aprendizaje de máscara
            "mask_out":  combined_seg_loss(),
            "class_out": "categorical_crossentropy"
        },
        loss_weights={"mask_out": 1.5, "class_out": 0.3},
        metrics={
            "mask_out":  [UpdatedMeanIoU(num_classes=NUM_CLASSES, name="miou"),
                          "accuracy"],
            "class_out": "accuracy"
        }
    )


# ============================================================================
# 5. DATOS
# ============================================================================
print("Cargando datasets...")
train_ds = BellPepperDataset(
    base_path + "train",
    base_path + "train/_annotations.coco.json",
    batch_size=BATCH_SIZE, augment=True)

val_ds = BellPepperDataset(
    base_path + "valid",
    base_path + "valid/_annotations.coco.json",
    batch_size=BATCH_SIZE, augment=False)

test_ds = BellPepperDataset(
    base_path + "test",
    base_path + "test/_annotations.coco.json",
    batch_size=BATCH_SIZE, augment=False)

# ============================================================================
# 6. ENTRENAMIENTO EN DOS FASES
# ============================================================================
model = build_mask_mobilenet_fixed()
model.summary()

# ── FASE 1: backbone congelado, solo decoder + cabeza aprenden ───────────────
print("\n" + "="*50)
print("FASE 1: Entrenando decoder con backbone congelado")
print("="*50)
compile_model(model, lr=1e-3)
h1 = model.fit(
    train_ds, validation_data=val_ds, epochs=30,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True,
                      monitor="val_mask_out_miou", mode="max"),
        ModelCheckpoint("best_pepper_fase1.keras", save_best_only=True,
                        monitor="val_mask_out_miou", mode="max"),
        ReduceLROnPlateau(factor=0.5, patience=4,
                          monitor="val_mask_out_miou", mode="max")
    ]
)

# ── FASE 2: descongelar solo las últimas capas del backbone ──────────────────
# Capa de resize serializable — reemplaza layers.Lambda
class BilinearResize(tf.keras.layers.Layer):
    def call(self, inputs):
        x, ref = inputs
        return tf.image.resize(x, tf.shape(ref)[1:3], method="bilinear")
    
    def get_config(self):
        return super().get_config()

def decoder_block(inputs, skip, filters):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    x = BilinearResize()([x, skip])  # ← sin Lambda
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    return x
print("\n" + "="*50)
print("FASE 2: Fine-tuning últimas capas del backbone")
print("="*50)

# Descongelar solo desde block_13 en adelante (las más específicas)
model.trainable = True
fine_tune_from = "block_13_expand"
set_trainable  = False
for layer in model.layers:
    if layer.name == fine_tune_from:
        set_trainable = True
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False   # BN siempre congelado
    else:
        layer.trainable = set_trainable

import builtins
trainable_count = builtins.sum(1 for l in model.layers if l.trainable)
print(f"  Capas entrenables en Fase 2: {trainable_count}")

compile_model(model, lr=1e-5)  # LR muy bajo para fine-tuning
h2 = model.fit(
    train_ds, validation_data=val_ds, epochs=20,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True,
                      monitor="val_mask_out_miou", mode="max"),
        ModelCheckpoint("best_pepper_fase2.keras", save_best_only=True,
                        monitor="val_mask_out_miou", mode="max"),
    ]
)

# ============================================================================
# 7. EVALUACIÓN
# ============================================================================
print("\n📦 Evaluando en test...")
# Cargar el mejor modelo de fase 2
import keras
keras.config.enable_unsafe_deserialization()

model = tf.keras.models.load_model(
    "best_pepper_fase2.keras",
    custom_objects={
        "UpdatedMeanIoU": UpdatedMeanIoU,
        "BilinearResize": BilinearResize,
        "loss": combined_seg_loss()
    }
)

results = model.evaluate(test_ds, verbose=0)
y_true_list, y_pred_list = [], []
for i in range(len(test_ds)):
    X, y = test_ds[i]
    preds = model.predict(X, verbose=0)
    y_true_list.extend(np.argmax(y['class_out'], axis=1))
    y_pred_list.extend(np.argmax(preds[1], axis=1))

_, recall, f1, _ = precision_recall_fscore_support(
    y_true_list, y_pred_list, average='weighted')

print("\n" + "="*50 + "\n🎯 RESULTADOS FINALES\n" + "="*50)
print(f"📉 Pérdida Total:          {results[0]:.4f}")
print(f"🎨 Segmentación IoU:  {results[5]*100:.2f}%")
print(f"🎨 Segmentación Acc:  {results[4]*100:.2f}%")
print(f"🏷️  Clasificación Acc: {results[3]*100:.2f}%")
print(f"🏷️  Recall:            {recall*100:.2f}%")
print(f"🏷️  F1:                {f1*100:.2f}%")
print("="*50)

# ============================================================================
# 8. DIAGNÓSTICO RÁPIDO POST-ENTRENAMIENTO
#    Verifica que la máscara ya no sea "todo BacterialSpot"
# ============================================================================
print("\n🔍 Verificando calidad de máscara post-entrenamiento...")
X_test, y_test = test_ds[0]
preds_test = model.predict(X_test, verbose=0)
mask_sample = preds_test[0][0]  # primera imagen, máscara (256,256,2)

canal_0 = mask_sample[:, :, 0]
canal_1 = mask_sample[:, :, 1]
argmax  = np.argmax(mask_sample, axis=-1)

print(f"  Canal 0 (BacterialSpot) — min={canal_0.min():.3f}  max={canal_0.max():.3f}  mean={canal_0.mean():.3f}")
print(f"  Canal 1 (Healthy)       — min={canal_1.min():.3f}  max={canal_1.max():.3f}  mean={canal_1.mean():.3f}")
print(f"  Argmax px BacterialSpot: {(argmax==0).sum()}  ({(argmax==0).mean()*100:.1f}%)")
print(f"  Argmax px Healthy:       {(argmax==1).sum()}  ({(argmax==1).mean()*100:.1f}%)")

if (argmax==0).mean() > 0.95:
    print("\n  ⚠️  Máscara aún predice todo como BacterialSpot")
    print("     → Aumentar épocas de Fase 1 o revisar balance del dataset")
else:
    print("\n  ✅ Máscara correcta — hay variación espacial")

# ============================================================================
# 9. EXPORT TFLITE
# ============================================================================
print("\n═══ Exportando a TFLite ═══")
model.save("PepMaskMobilnet_v2.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
with open("PepMaskMobilnet_v2.tflite", "wb") as f:
    f.write(tflite_model)

print(f"✅ PepMaskMobilnet_v2.tflite — {len(tflite_model)/1024/1024:.2f} MB")

# ============================================================================
# 10. LATENCIA
# ============================================================================
total_params = model.count_params() / 1e6
sample_img = np.expand_dims(np.zeros((256, 256, 3)), 0)
for _ in range(10): model.predict(sample_img, verbose=0)

start = time.time()
for _ in range(100): model.predict(sample_img, verbose=0)
avg_ms = ((time.time() - start) / 100) * 1000

print(f"\nParámetros: {total_params:.2f}M")
print(f"Inferencia: {avg_ms:.2f} ms")

# ============================================================================
# 11. VISUALIZACIÓN
# ============================================================================
cm = confusion_matrix(y_true_list, y_pred_list)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(6, 5))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (%) — Bell Pepper v2')
plt.savefig('cm_pepper_v2.png')
plt.show()

# Historial combinado
all_history = {}
for k in h2.history:
    all_history[k] = h1.history.get(k, []) + h2.history[k]

pd.DataFrame(all_history).to_csv("history_pepper_v2.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (train_k, val_k, title) in zip(axes, [
    ('loss',               'val_loss',               'Total Loss'),
    ('mask_out_miou',      'val_mask_out_miou',      'Segmentation IoU'),
    ('class_out_accuracy', 'val_class_out_accuracy', 'Classification Acc'),
]):
    ax.plot(all_history.get(train_k, []), label='Train')
    ax.plot(all_history.get(val_k,   []), label='Val')
    ax.set_title(title); ax.legend(); ax.grid(True)
    # Línea vertical separando fase 1 y 2
    ax.axvline(len(h1.history['loss']), color='red',
               linestyle='--', alpha=0.5, label='Fase 1→2')

plt.tight_layout()
plt.savefig('history_pepper_v2.png', dpi=150)
plt.show()