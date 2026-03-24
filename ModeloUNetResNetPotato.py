# -*- coding: utf-8 -*-
"""
SOLANAPP: Modelo de Segmentación y Clasificación Multi-Output
Configuración: 3 Clases (EarlyBlight, Healthy, LateBlight)
Basado en Arquitectura ResNet50-Unet
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
# 2. DATASET CON MAPEO A 3 CLASES
# ============================================================================
class PotatoDiseaseDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Mapeo estratégico: Convertimos las 6 categorías originales en 3
        # 0, 2 -> EarlyBlight (0)
        # 4    -> LateBlight (2)
        # 3, 1, 5 -> Healthy o background (1)
        self.cat_map = {
            0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 1
        }
        self.healthy_idx = 1 

    def __len__(self):
        return len(self.ids) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            
            if image is None: continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)

            mask = np.zeros((self.img_size[0], self.img_size[1], NUM_CLASSES), dtype=np.float32)
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            anns = self.coco.loadAnns(ann_ids)
            
            current_class_idx = self.healthy_idx
            
            if anns:
                first_cat_id = anns[0]['category_id']
                current_class_idx = self.cat_map.get(first_cat_id, self.healthy_idx)
                
                for ann in anns:
                    mapped_id = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= mapped_id < NUM_CLASSES:
                        if 'segmentation' in ann and ann['segmentation']:
                            m = self.coco.annToMask(ann)
                            m_res = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, mapped_id] = np.maximum(mask[:, :, mapped_id], m_res.astype(np.float32))
                
                # Complemento: Píxeles sin enfermedad son Healthy (Canal 1)
                mask[:, :, self.healthy_idx] = (np.max(mask, axis=-1) == 0).astype(np.float32)
            else:
                mask[:, :, self.healthy_idx] = 1.0
            
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=NUM_CLASSES))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

# ============================================================================
# 3. ARQUITECTURA HÍBRIDA (U-NET + RESNET50)
# ============================================================================

def build_model(input_shape=(256, 256, 3), num_classes=3):
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False 

    # Capas de salto (Skip Connections)
    s1, s2, s3, s4 = base_model.input, base_model.get_layer("conv1_relu").output, \
                     base_model.get_layer("conv2_block3_out").output, base_model.get_layer("conv3_block4_out").output
    bridge = base_model.output 

    def decoder_block(inputs, skip, filters):
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
        x = layers.Resizing(skip.shape[1], skip.shape[2])(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    d1 = decoder_block(bridge, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(d4)
    
    # Cabeza de Clasificación
    gap = layers.GlobalAveragePooling2D()(bridge)
    fc = layers.Dense(256, activation='relu')(gap)
    fc = layers.Dropout(0.4)(fc)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(fc)

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# ============================================================================
# 4. PÉRDIDAS Y MÉTRICAS
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

# ============================================================================
# 5. COMPILACIÓN Y ENTRENAMIENTO
# ============================================================================
model = build_model(num_classes=NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss_weights={"mask_out": 1.0, "class_out": 0.5}, # Balanceamos según vimos en tus dudas anteriores

    loss={"mask_out": focal_tversky_loss, "class_out": "categorical_crossentropy"},
    metrics={"mask_out": [UpdatedMeanIoU(num_classes=NUM_CLASSES, name="miou"), "accuracy"], "class_out": "accuracy"}
)

# Cargar Datasets (Cambia las rutas a las tuyas)
path = 'D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v5i.coco/'
train_ds = PotatoDiseaseDataset(path+"train", path+"train/_annotations.coco.json")
val_ds = PotatoDiseaseDataset(path+"valid", path+"valid/_annotations.coco.json")
test_ds = PotatoDiseaseDataset(path + "test", path + "test/_annotations.coco.json")


print(f"🚀 Iniciando entrenamiento para {NUM_CLASSES} clases...")
history = model.fit(train_ds, validation_data=val_ds, epochs=50, 
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                               ModelCheckpoint('best_model_3classes.keras', save_best_only=True)])




import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("history_unet_resnetPotato.csv", index=False)


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
# 4. FUNCIÓN DE EVALUACIÓN FINAL (Recall, F1, IoU)
# ============================================================================

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

import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Matriz de Confusión Normalizada
cm = confusion_matrix(y_true, y_pred)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (%)Potato - ResNet+ Unet')
plt.savefig('final_cm.png')
plt.show()

# ============================================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['mask_out_accuracy'], label='Train Mask Acc')
plt.plot(history.history['val_mask_out_accuracy'], label='Val Mask Acc')
plt.title('Segmentation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['class_out_accuracy'], label='Train Class Acc')
plt.plot(history.history['val_class_out_accuracy'], label='Val Class Acc')
plt.title('Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history_5classes.png', dpi=150)
plt.show()

print("\n🎨 Generando visualizaciones...")




import matplotlib.pyplot as plt
import numpy as np

# 1. VERIFICACIÓN CRÍTICA DE DATOS
try:
    data = test_ds[0]
    X_batch, y_batch = data
    print(f"✅ X_batch shape: {X_batch.shape}")
    print(f"✅ y_batch keys: {y_batch.keys()}")
except Exception as e:
    print(f"❌ Error al extraer datos del dataset: {e}")
    # Forzar una salida si los datos están vacíos
    X_batch = np.array([]) 

# 1. Obtener predicciones (Ya sabemos que el batch tiene 8 imágenes)
predictions = model.predict(X_batch, verbose=0)
mask_preds = predictions[0]   # (8, 256, 256, 5)
class_preds = predictions[1]  # (8, 5)

class_names = ["EarlyBlight", "Healthy", "LateBlight"]
colors = ['Reds',  'Greens', 'Blues']

# 2. Configurar la cuadrícula (Mostraremos 4 ejemplos del batch de 8)
num_samples = 4 
fig, axes = plt.subplots(num_samples, 4, figsize=(18, num_samples * 4))

for i in range(num_samples):
    # --- PROCESAR IMAGEN ---
    # Desnormalización simple para ver los colores reales
    img_vis = X_batch[i]
    if img_vis.max() <= 1.0: # Si está en rango [0,1]
        img_vis = (img_vis * 255).astype(np.uint8)
    else: # Si ya está en [0,255] o preprocesada
        img_vis = ((img_vis - img_vis.min()) / (img_vis.max() - img_vis.min()) * 255).astype(np.uint8)
    
    # --- DATOS REALES (GT) ---
    gt_mask = np.argmax(y_batch['mask_out'][i], axis=-1)
    gt_class_idx = np.argmax(y_batch['class_out'][i])
    
    # --- PREDICCIONES ---
    pred_mask_all = np.argmax(mask_preds[i], axis=-1)
    pred_class_idx = np.argmax(class_preds[i])
    pred_conf = class_preds[i][pred_class_idx]
    
    # COLUMNA 0: Imagen Original + Etiqueta Real
    axes[i, 0].imshow(img_vis)
    axes[i, 0].set_title(f"REAL: {class_names[gt_class_idx]}", fontsize=10, color='blue')
    axes[i, 0].axis('off')
    
    # COLUMNA 1: Máscara Real (Ground Truth)
    axes[i, 1].imshow(gt_mask, cmap='viridis', vmin=0, vmax=4)
    axes[i, 1].set_title("Máscara Real", fontsize=10)
    axes[i, 1].axis('off')
    
    # COLUMNA 2: Máscara Predicha (Segmentación)
    axes[i, 2].imshow(pred_mask_all, cmap='viridis', vmin=0, vmax=4)
    axes[i, 2].set_title(f"Segmentación Predicha", fontsize=10)
    axes[i, 2].axis('off')
    
    # COLUMNA 3: Confianza de Clasificación
    # Mostramos el canal de la enfermedad que el modelo cree que tiene
    heatmap = mask_preds[i, :, :, pred_class_idx]
    axes[i, 3].imshow(heatmap, cmap=colors[pred_class_idx])
    axes[i, 3].set_title(f"PRED: {class_names[pred_class_idx]}\nConf: {pred_conf:.2f}", 
                         fontsize=10, fontweight='bold', color='red')
    axes[i, 3].axis('off')

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()

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
print()

# ============================================================================
# GUARDAR MODELO
# ============================================================================
print("\n💾 Guardando modelo...")

model.save("PotatoUnetResNet2.keras")
print("✅ Modelo guardado: PotatoUnetResNet2.keras")

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

with open('PotatoUnetResNet2.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: PotatoUnetResNet2.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")


# ... (Continuar con la carga de datasets y model.fit igual que antes)