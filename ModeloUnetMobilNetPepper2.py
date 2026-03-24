# -*- coding: utf-8 -*-
"""
MobileNetV2-UNet OPTIMIZADO: BELL PEPPER EDITION (Single Phase Training)
Categorías: BacterialSpot (0), Healthy (1)
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ============================================================================
# 1. CONFIGURACIÓN GLOBAL
# ============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
CLASS_NAMES = ["BacterialSpot", "Healthy"]
NUM_CLASSES = len(CLASS_NAMES)
base_path   = "D:/DATASETS/Imagenes/Solanaceas/BellPepper.v1i.coco/"

# ============================================================================
# 2. DATASET
# ============================================================================
class BellPepperDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256), augment=False):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.cat_map = {0: 0, 1: 0, 2: 1} # 0,1 -> BacterialSpot, 2 -> Healthy
        self.healthy_idx = 1

    def __len__(self):
        return len(self.ids) // self.batch_size

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            image = cv2.imread(os.path.join(self.img_dir, img_info['file_name']))
            if image is None: continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            mask_single = np.zeros(self.img_size, dtype=np.uint8)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            current_class_idx = self.healthy_idx
            if anns:
                mapped_ids = [self.cat_map.get(a['category_id'], self.healthy_idx) for a in anns]
                current_class_idx = 0 if 0 in mapped_ids else 1
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= m_id < NUM_CLASSES:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask_single[m.astype(bool)] = m_id
                        except: continue
            
            if mask_single.max() == 0 and not anns:
                mask_single[:] = self.healthy_idx

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(tf.keras.utils.to_categorical(mask_single, num_classes=NUM_CLASSES))
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=NUM_CLASSES))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

# ============================================================================
# 3. BLOQUES DE ARQUITECTURA
# ============================================================================
def _dw_conv_bn_relu(x, filters):
    x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    x = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    return x

def _aspp_block(x, filters=128):
    b0 = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.ReLU(6.0)(b0)
    
    # Simple branch GAP
    gap = layers.GlobalAveragePooling2D()(x)
    gap = layers.Reshape((1, 1, x.shape[-1]))(gap)
    gap = layers.Conv2D(filters, (1, 1), use_bias=False)(gap)
    gap = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation="bilinear")(gap)

    x = layers.Concatenate()([b0, gap])
    x = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU(6.0)(x)

def _decoder_block(inputs, skip, filters):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    x = layers.Concatenate()([x, skip])
    return _dw_conv_bn_relu(x, filters)

# ============================================================================
# 4. CONSTRUCCIÓN DEL MODELO
# ============================================================================
def build_model(input_shape=(256, 256, 3)):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = True # ACTIVADO PARA UNA SOLA FASE

    # Skip connections
    s4 = base.get_layer("block_6_expand_relu").output  # 32x32
    s3 = base.get_layer("block_3_expand_relu").output  # 64x64
    s2 = base.get_layer("block_1_expand_relu").output  # 128x128
    bridge = base.get_layer("out_relu").output         # 8x8

    x = _aspp_block(bridge, 128)
    x = _decoder_block(x, base.get_layer("block_13_expand_relu").output, 128) # 16x16
    x = _decoder_block(x, s4, 64)  # 32x32
    x = _decoder_block(x, s3, 32)  # 64x64
    x = _decoder_block(x, s2, 16)  # 128x128
    
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x) # 256x256
    mask_out = layers.Conv2D(NUM_CLASSES, (1, 1), activation="softmax", name="mask_out")(x)

    # Classification head
    gap = layers.GlobalAveragePooling2D()(bridge)
    class_out = layers.Dense(NUM_CLASSES, activation="softmax", name="class_out")(layers.Dropout(0.4)(gap))

    return models.Model(inputs=base.input, outputs=[mask_out, class_out])

# ============================================================================
# 5. PÉRDIDAS Y MÉTRICAS
# ============================================================================
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.33):
    y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
    tvi = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
    return tf.reduce_mean(tf.pow(1.0 - tvi, gamma))

# ============================================================================
# 6. ENTRENAMIENTO ÚNICO
# ============================================================================
train_ds = BellPepperDataset(base_path + "train", base_path + "train/_annotations.coco.json")
val_ds   = BellPepperDataset(base_path + "valid", base_path + "valid/_annotations.coco.json")
test_ds = BellPepperDataset(
    base_path + "test",
    base_path + "test/_annotations.coco.json",
    batch_size=BATCH_SIZE, augment=False)


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # LR moderado para fase única
    loss={"mask_out": focal_tversky_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 2.0, "class_out": 1.0},
    metrics={
        "mask_out": [UpdatedMeanIoU(num_classes=NUM_CLASSES, name="miou"), "accuracy"],
        "class_out": "accuracy"
    }
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint("best_pepper_single_phase.keras", save_best_only=True)
]

print("🚀 Iniciando Entrenamiento Único...")
history = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=callbacks)

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
print("\n📦 Evaluando resultados finales...")
results = model.evaluate(test_ds, verbose=0)

y_true_list, y_pred_list = [], []
for i in range(len(test_ds)):
    X, y = test_ds[i]
    preds = model.predict(X, verbose=0)
    y_true_list.extend(np.argmax(y['class_out'], axis=1))
    y_pred_list.extend(np.argmax(preds[1], axis=1))

# Métricas extra
_, recall, f1, _ = precision_recall_fscore_support(y_true_list, y_pred_list, average='weighted')

print("\n" + "="*50 + "\n🎯 RESULTADOS FINALES\n" + "="*50)
print(f"📉 Pérdida Total:               {results[0]:.4f}")
print(f"---")
print(f"🎨 Segmentación IoU:      {results[5]*100:.2f}%")
print(f"🎨 Segmentación Acc:      {results[4]*100:.2f}%")
print(f"🏷️  Clasificación Acc:     {results[3]*100:.2f}%")
print(f"🏷️  Clasificación Recall:  {recall*100:.2f}%")
print(f"🏷️  Clasificación F1:      {f1*100:.2f}%")
print("="*50)


import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("history_unet_mobilenet_pepper2.csv", index=False)


# ============================================================================
# 7. EXPORTACIÓN Y VISUALIZACIÓN
# ============================================================================
model.save("PepUnetMobilnet2.keras")

# Matriz de Confusión
test_ds = BellPepperDataset(base_path + "test", base_path + "test/_annotations.coco.json")
y_true_all, y_pred_all = [], []
for i in range(len(test_ds)):
    X, y = test_ds[i]
    p = model.predict(X, verbose=0)
    y_true_all.extend(np.argmax(y['class_out'], axis=1))
    y_pred_all.extend(np.argmax(p[1], axis=1))

cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
plt.title("Matriz de Confusión - Pimiento")
plt.show()

# Visualización de 1 ejemplo
X_sample, y_sample = test_ds[0]
preds = model.predict(X_sample, verbose=0)
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1); plt.imshow(X_sample[0]*0.5 + 0.5); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(np.argmax(y_sample['mask_out'][0], -1)); plt.title("Real")
plt.subplot(1,3,3); plt.imshow(np.argmax(preds[0][0], -1)); plt.title("Predicción")
plt.show()

model.save("PepUnetMobilnet2.keras")

# Conversión a TFLite con float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
# Fallback para ops no soportadas
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()
tflite_path  = "PepUnetMobilnet2.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: PepUnetMobilnet2.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

