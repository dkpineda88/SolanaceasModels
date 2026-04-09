# -*- coding: utf-8 -*-
"""
SOLANAPP: Modelo de Segmentación y Clasificación (3 Clases)
Backbone: ResNet50
Clases: EarlyBlight, Healthy, LateBlight
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ============================================================================
# 1. PARÁMETROS GLOBALES
# ============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_CLASSES = 3  # 0: EarlyBlight, 1: Healthy, 2: LateBlight
CLASS_NAMES = ["EarlyBlight", "Healthy", "LateBlight"]

# ============================================================================
# 2. DATASET PERSONALIZADO (Mapeo a 3 Clases)
# ============================================================================
class PotatoDiseaseDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = NUM_CLASSES
        self.healthy_idx = 1
        
        # Mapeo estratégico: IDs COCO originales -> Nuevas 3 clases
        # 0, 2: EarlyBlight (0) | 1, 3, 5: Healthy (1) | 4: LateBlight (2)
        self.cat_map = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 1}

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

            mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            anns = self.coco.loadAnns(ann_ids)
            
            current_class_idx = self.healthy_idx
            if anns:
                # La clase de la imagen se define por la primera anotación mapeada
                current_class_idx = self.cat_map.get(anns[0]['category_id'], self.healthy_idx)
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= m_id < self.num_classes and 'segmentation' in ann:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, m_id] = np.maximum(mask[:, :, m_id], m.astype(np.float32))
                        except: continue
            
            # Píxeles sin enfermedad = Healthy (Canal 1)
            mask[:, :, self.healthy_idx] = (np.max(mask, axis=-1) == 0).astype(np.float32)
            
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=self.num_classes))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

# ============================================================================
# 3. ARQUITECTURA RESNET50 MULTI-OUTPUT
# ============================================================================

def build_resnet_3classes(input_shape=(256, 256, 3), num_classes=3):
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False 

    # Skip connections
    s1 = base_model.input # 256
    s2 = base_model.get_layer("conv1_relu").output # 128
    s3 = base_model.get_layer("conv2_block3_out").output # 64
    s4 = base_model.get_layer("conv3_block4_out").output # 32

    # Bridge
    bridge = base_model.get_layer("conv5_block3_out").output # 8x8

    # Mask Head (Decoder)
    x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(bridge) # 16
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(x) # 32
    x = layers.Concatenate()([x, s4])
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(x) # 64
    x = layers.Concatenate()([x, s3])
    x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x) # 256
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(x)

    # Classification Head
    gap = layers.GlobalAveragePooling2D()(bridge)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(layers.Dropout(0.4)(layers.Dense(512, activation='relu')(gap)))

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# ============================================================================
# 4. COMPILACIÓN Y MÉTRICAS
# ============================================================================
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
    tversky = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
    return tf.reduce_mean(tf.pow((1 - tversky), gamma))

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

model = build_resnet_3classes()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"mask_out": focal_tversky_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.5, "class_out": 0.5},
    metrics={"mask_out": [UpdatedMeanIoU(num_classes=3, name="miou"), "accuracy"], "class_out": "accuracy"}
)

# ============================================================================
# 5. CARGA DE DATOS Y ENTRENAMIENTO
# ============================================================================
# Ajustar rutas según tu sistema
base_path = 'D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v5i.coco/'
train_ds = PotatoDiseaseDataset(base_path+"train", base_path+"train/_annotations.coco.json")
val_ds = PotatoDiseaseDataset(base_path+"valid", base_path+"valid/_annotations.coco.json")
test_ds = PotatoDiseaseDataset(base_path+"test", base_path+"test/_annotations.coco.json")

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_tomato_3class.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=3)
]

print("\n🚀 Entrenando para 3 clases...")
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)


import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("history_mask_resnetPotato.csv", index=False)

print("\n📦 Evaluando modelo...")
res = model.evaluate(test_ds, verbose=0)
y_true, y_pred = [], []
for i in range(len(test_ds)):
    X, y = test_ds[i]
    p = model.predict(X, verbose=0)
    y_true.extend(np.argmax(y['class_out'], axis=1))
    y_pred.extend(np.argmax(p[1], axis=1))

_, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

print("\n" + "="*50 + "\n🎯 RESULTADOS FINALES\n" + "="*50)
print(f"📉 Pérdida Total:               {res[0]:.4f}")
print(f"---")
print(f"🎨 Segmentación IoU:      {res[5]*100:.2f}%")
print(f"🎨 Segmentación Acc:      {res[4]*100:.2f}%")
print(f"🏷️  Clasificación Acc:     {res[3]*100:.2f}%")
print(f"🏷️  Clasificación Recall:  {recall*100:.2f}%")
print(f"🏷️  Clasificación F1:      {f1*100:.2f}%")
print("="*50)

import time

# 1. Contar parámetros (en millones)
total_params = model.count_params() / 1e6

# 2. Medir tiempo de inferencia (promedio de 100 predicciones)
sample_img = np.expand_dims(np.zeros((256, 256, 3)), 0)
for _ in range(10): model.predict(sample_img, verbose=0) # Warm-up

start_time = time.time()
for _ in range(100):
    model.predict(sample_img, verbose=0)
avg_inference_ms = ((time.time() - start_time) / 100) * 1000

print(f"Parámetros: {total_params:.2f}M")
print(f"Inferencia: {avg_inference_ms:.2f} ms")

# ============================================================================
# LATENCY
# ============================================================================
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

# Uso:
latency = calculate_latency(model, test_ds)

# ============================================================================
# 6. GRÁFICAS DE HISTORIAL
# ============================================================================
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Total Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['mask_out_miou'], label='Train IoU')
plt.plot(history.history['val_mask_out_miou'], label='Val IoU')
plt.title('Segmentation mIoU')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['class_out_accuracy'], label='Train Acc')
plt.plot(history.history['val_class_out_accuracy'], label='Val Acc')
plt.title('Classification Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# 7. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
# ============================================================================
print("\n📊 Generando Matriz de Confusión...")
y_true_list, y_pred_list = [], []
for i in range(len(test_ds)):
    X, y = test_ds[i]
    preds = model.predict(X, verbose=0)
    y_true_list.extend(np.argmax(y['class_out'], axis=1))
    y_pred_list.extend(np.argmax(preds[1], axis=1))

cm = confusion_matrix(y_true_list, y_pred_list)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Confusion Matrix - 3 Classes')
plt.show()

# ============================================================================
# 8. VISUALIZACIÓN DE PREDICCIONES (OUT)
# ============================================================================
X_batch, y_batch = test_ds[0]
preds = model.predict(X_batch, verbose=0)
colors = ['Reds', 'Greens', 'Blues']

fig, axes = plt.subplots(4, 3, figsize=(12, 12))
for i in range(4):
    # Imagen Original
    img = X_batch[i]
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Real: {CLASS_NAMES[np.argmax(y_batch['class_out'][i])]}")
    axes[i, 0].axis('off')

    # Máscara Real
    axes[i, 1].imshow(np.argmax(y_batch['mask_out'][i], axis=-1), cmap='viridis')
    axes[i, 1].set_title("GT Mask")
    axes[i, 1].axis('off')

    # Máscara Predicha
    pred_idx = np.argmax(preds[1][i])
    axes[i, 2].imshow(preds[0][i, :, :, pred_idx], cmap=colors[pred_idx])
    axes[i, 2].set_title(f"Pred: {CLASS_NAMES[pred_idx]} ({preds[1][i, pred_idx]:.2f})")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# 9. EXPORTACIÓN TFLITE
# ============================================================================
model.save("PotatoMaskResNet.keras")
print("✅ Modelo guardado: PotatoMaskResNet.keras")

# TFLite
print("\n🔄 Convirtiendo a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
converter.target_spec.supported_types = [tf.float32]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open('PotatoMaskResNet.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: PotatoMaskResNet.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")
