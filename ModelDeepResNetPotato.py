# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:49:44 2026
@author: dkpin
Modelo: DeepLabV3+ ResNet50 para 3 Clases (EarlyBlight, Healthy, LateBlight)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
CLASS_NAMES = ["EarlyBlight", "Healthy", "LateBlight"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# 1. DATASET CUSTOM (3 CLASES)
# ============================================================================
# ============================================================================
# 1. DATASET CON PROTECCIÓN ANTI-INDEXERROR
# ============================================================================
class PotatoDataset3(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = NUM_CLASSES
        
        # MAPEO DINÁMICO: Asegúrate de que estos IDs coincidan con tu JSON
        # 0: EarlyBlight, 1: Healthy, 2: LateBlight
        self.cat_map = {0: 0, 1: 0, 2: 1, 3: 2} # Ejemplo: fusiona ID 0 y 1 en EarlyBlight
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

            mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            anns = self.coco.loadAnns(ann_ids)
            
            current_class_idx = self.healthy_idx # Default: Healthy
            
            if anns:
                # Determinar clase principal para clasificación
                first_cat = anns[0]['category_id']
                current_class_idx = self.cat_map.get(first_cat, self.healthy_idx)
                
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    # VALIDACIÓN CRÍTICA: Evita el IndexError
                    if (0 <= m_id < self.num_classes and 
                        'segmentation' in ann and 
                        isinstance(ann['segmentation'], list) and 
                        len(ann['segmentation']) > 0):
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, m_id] = np.maximum(mask[:, :, m_id], m.astype(np.float32))
                        except Exception:
                            continue # Salta anotaciones corruptas
            
            # Generar canal Healthy (donde no hay Early ni Late)
            disease_presence = np.max(mask[:, :, [0, 2]], axis=-1)
            mask[:, :, self.healthy_idx] = (disease_presence == 0).astype(np.float32)
            
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=self.num_classes))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

# ============================================================================
# 2. ARQUITECTURA DEEPLABV3+ (RESNET50)
# ============================================================================
def AtrousSpatialPyramidPooling(tensor):
    dims = tensor.shape
    # Convoluciones paralelas con diferentes dilation rates
    out_1 = layers.Conv2D(256, 1, padding="same", use_bias=False)(tensor)
    out_6 = layers.Conv2D(256, 3, dilation_rate=6, padding="same", use_bias=False)(tensor)
    out_12 = layers.Conv2D(256, 3, dilation_rate=12, padding="same", use_bias=False)(tensor)
    
    # Image Pooling
    out_pool = layers.GlobalAveragePooling2D()(tensor)
    out_pool = layers.Reshape((1, 1, dims[-1]))(out_pool)
    out_pool = layers.Conv2D(256, 1, padding="same", use_bias=False)(out_pool)
    out_pool = layers.UpSampling2D(size=(dims[1], dims[2]), interpolation="bilinear")(out_pool)
    
    res = layers.Concatenate()([out_1, out_6, out_12, out_pool])
    res = layers.Conv2D(256, 1, padding="same", use_bias=False)(res)
    res = layers.BatchNormalization()(res)
    res = layers.Activation("relu")(res)
    return res

def build_model():
    base = tf.keras.applications.ResNet50(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    
    # Branch Segmentación (DeepLab Head)
    res_out = base.get_layer("conv4_block6_out").output 
    aspp = AtrousSpatialPyramidPooling(res_out)
    x = layers.UpSampling2D(size=(16, 16), interpolation="bilinear")(aspp)
    mask_out = layers.Conv2D(NUM_CLASSES, 1, activation='softmax', name="mask_out")(x)
    
    # Branch Clasificación
    c5 = base.output
    gap = layers.GlobalAveragePooling2D()(c5)
    drop = layers.Dropout(0.4)(gap)
    class_out = layers.Dense(NUM_CLASSES, activation='softmax', name="class_out")(drop)

    return models.Model(inputs=base.input, outputs=[mask_out, class_out])
# ============================================================================
# 3. COMPILACIÓN Y ENTRENAMIENTO
# ============================================================================
model = build_model()

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    # y_true y y_pred: (batch, h, w, classes)
    y_true_f = tf.cast(tf.reshape(y_true, [-1, 3]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, 3]), tf.float32)
    
    # Cálculo de Tversky Index
    # TP (True Positives)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    # FN (False Negatives) - Píxeles de enfermedad ignorados
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
    # FP (False Positives) - Ruido predicho como enfermedad
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
    
    # Alfa alto (0.7) penaliza más los FNs (sensibilidad)
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    
    # Aplicar componente Focal para ejemplos difíciles
    ft_loss = tf.pow((1 - tversky_index), 1/gamma)
    
    return tf.reduce_mean(ft_loss)

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertimos las probabilidades (softmax) y el One-Hot a índices enteros
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# En tu compilación:
iou_metric = UpdatedMeanIoU(num_classes=3, name="mean_iou")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"mask_out": focal_tversky_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.5, "class_out": 0.5},
    metrics={
        "mask_out": [iou_metric, "accuracy"],
        "class_out": "accuracy"
    }
)

# --- Rutas de datos ---
base_path = 'D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v5i.coco/'
train_ds = PotatoDataset3(base_path + "train", base_path + "train/_annotations.coco.json")
val_ds = PotatoDataset3(base_path + "valid", base_path + "valid/_annotations.coco.json")
test_ds = PotatoDataset3(base_path + "test", base_path + "test/_annotations.coco.json")

# --- Fit ---
history = model.fit(
    train_ds, validation_data=val_ds, epochs=50, 
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True), 
               ModelCheckpoint('best_3class.keras', save_best_only=True)]
)

import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("history_deeplabv3_resnetPotato.csv", index=False)


# ============================================================================
# 5. para grafico de dispersion
# ============================================================================


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
# 6. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
# ============================================================================
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

# Matriz de Confusión Normalizada
cm = confusion_matrix(y_true_list, y_pred_list)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (%)Potato - DeepLabV3+ ResNet')
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
plt.savefig('training_history_3classes.png', dpi=150)
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
colors = ['Reds', 'Greens', 'Blues']

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
    axes[i, 1].set_title("Ground Truth", fontsize=10)
    axes[i, 1].axis('off')
    
    # COLUMNA 2: Máscara Predicha (Segmentación)
    axes[i, 2].imshow(pred_mask_all, cmap='viridis', vmin=0, vmax=4)
    axes[i, 2].set_title(f"Predicted Segmentation", fontsize=10)
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
# Guardar Modelo
model.save("PotatoDeepResNet_3Class.keras")

print("\n🔄 Convirtiendo a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
converter.target_spec.supported_types = [tf.float32]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open('PotatoDeepResNet_3Class.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: PotatoDeepResNet_3Class.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

print("✅ Proceso completado.")