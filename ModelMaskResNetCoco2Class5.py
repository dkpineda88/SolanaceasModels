# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:49:44 2026

@author: dkpin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 08:22:34 2026

@author: dkpin
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_CLASSES = 5  # 0:EarlyBlight(fusionado), 1:BacterialSpot, 2:Healthy, 3:LateBlight, 4:YellowLeaf

# ============================================================================
# 1. DATASET (Corregido para evitar IndexError)
# ============================================================================
class TomatoDiseaseDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = 5
        self.healthy_idx = 2
        self.cat_map = {0: 0, 1: 1, 2: 0, 3: 2, 4: 3, 5: 4}
        self.class_names = ["EarlyBlight", "BacterialSpot", "Healthy", "LateBlight", "YellowLeafCurl"]

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
                current_class_idx = self.cat_map.get(anns[0]['category_id'], self.healthy_idx)
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    # VALIDACIÓN CRÍTICA para evitar IndexError en pycocotools
                    if 0 <= m_id < self.num_classes and 'segmentation' in ann and len(ann['segmentation']) > 0:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, m_id] = np.maximum(mask[:, :, m_id], m.astype(np.float32))
                        except:
                            continue # Si la anotación está mal, la saltamos
            
            mask[:, :, self.healthy_idx] = (np.max(mask, axis=-1) == 0).astype(np.float32)
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=self.num_classes))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

    def summarize_dataset(self):
        """Muestra la estructura: Clase + Cantidad"""
        counts = {name: 0 for name in self.class_names}
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            idx = self.cat_map.get(anns[0]['category_id'], self.healthy_idx) if anns else self.healthy_idx
            counts[self.class_names[idx]] += 1
        print("\n📊 RESUMEN DEL DATASET:")
        for k, v in counts.items(): print(f" - {k}: {v} imágenes")

# ============================================================================
# 2. ARQUITECTURA RESNET50 (Ajustada)
# ============================================================================
def build_mask_rcnn_resnet50(input_shape=(256, 256, 3), num_classes=5):
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False # Congelamos el backbone para estabilidad inicial

    c5 = base_model.get_layer("conv5_block3_out").output
    shared = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(c5)

    # Classification Head
    gap = layers.GlobalAveragePooling2D()(c5)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(layers.Dropout(0.4)(layers.Dense(1024, activation='relu')(gap)))

    # Mask Head (Decoder)
    x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same", activation="relu")(shared) # 16x16
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", activation="relu")(x) # 32x32
    x = layers.UpSampling2D((8, 8), interpolation='bilinear')(x) # 256x256
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(x)

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

model = build_mask_rcnn_resnet50()
# Crear el modelo
# ============================================================================
# 3. COMPILACIÓN Y MÉTRICAS
# ============================================================================
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertimos las probabilidades (softmax) y el One-Hot a índices enteros
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# En tu compilación:
iou_metric = UpdatedMeanIoU(num_classes=5, name="mean_iou")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # LR más bajo para ResNet
    loss={
        "mask_out": "categorical_crossentropy", 
        "class_out": "categorical_crossentropy"
    },
    loss_weights={
        "mask_out": 5.0, # Mask R-CNN requiere mucha fuerza en la máscara
        "class_out": 0.5
    },
    metrics={
        "mask_out": [iou_metric, "accuracy"],
        "class_out": "accuracy"
    }
)
print("\n📂 Cargando datasets...")

base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v5i.coco/'

train_ds = TomatoDiseaseDataset(
    base_path + "train", 
    base_path + "train/_annotations.coco.json",
    batch_size=BATCH_SIZE
)

val_ds = TomatoDiseaseDataset(
    base_path + "valid", 
    base_path + "valid/_annotations.coco.json",
    batch_size=BATCH_SIZE
)

test_ds = TomatoDiseaseDataset(
    base_path + "test", 
    base_path + "test/_annotations.coco.json",
    batch_size=BATCH_SIZE
)

print(f"✅ Train: {len(train_ds)} batches")
print(f"✅ Val: {len(val_ds)} batches")
print(f"✅ Test: {len(test_ds)} batches")

# ============================================================================
# VERIFICACIÓN DE DATOS

# ============================================================================
# CALLBACKS
# ============================================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_tomato_disease_5classes.keras', 
    monitor='val_class_out_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-7,
    verbose=1
)
# ============================================================================
# ENTRENAMIENTO
# ============================================================================
print("\n🚀 Iniciando entrenamiento...\n")

history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=50, 
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("history_maskrcnn_resnet.csv", index=False)





results = model.evaluate(test_ds)

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

class_names = ["EarlyBlight", "BacterialSpot", "Healthy", "LateBlight", "YellowLeafCurl"]
colors = ['Reds', 'Oranges', 'Greens', 'Blues', 'YlOrBr']

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
# 5. MATRIZ DE CONFUSIÓN
# ============================================================================
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("\n📊 Generando Matriz de Confusión...")

# 1. Obtener todas las predicciones y etiquetas reales del test_ds
y_true_all = []
y_pred_all = []

for i in range(len(test_ds)):
    X_batch, y_batch = test_ds[i]
    preds = model.predict(X_batch, verbose=0)
    
    # Extraemos los índices de las clases (argmax)
    y_true_all.extend(np.argmax(y_batch['class_out'], axis=1))
    y_pred_all.extend(np.argmax(preds[1], axis=1))

# 2. Calcular la matriz
# 5. MATRIZ DE CONFUSIÓN
# ============================================================================
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_true_all = []
y_pred_all = []

for i in range(len(test_ds)):
    X_batch, y_batch = test_ds[i]
    preds = model.predict(X_batch, verbose=0)
    
    # Extraemos los índices de las clases (argmax)
    y_true_all.extend(np.argmax(y_batch['class_out'], axis=1))
    y_pred_all.extend(np.argmax(preds[1], axis=1))
    
def plot_confusion_matrix_percent(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    # Normalización por fila (proporción de aciertos por clase real)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Normalized Confusion Matrix (%) Tomato - Mask+MobilNetV2', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig('confusion_matrix_percent.png', dpi=300)
    plt.show()

plot_confusion_matrix_percent(y_true_all, y_pred_all, class_names)

print("✅ Matriz de confusión guardada como: confusion_matrix_solanaceae.png")

# ============================================================================
# GUARDAR MODELO
# ============================================================================
print("\n💾 Guardando modelo...")

model.save("TMTMaskResNet5Class.keras")
print("✅ Modelo guardado: TMTMaskResNet5Class.keras")

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

with open('TMTMaskResNet5Class.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: TMTMaskResNet5Class.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
print("\n📋 RESUMEN DE CLASES:")
for i, name in enumerate(class_names):
    print(f"  Clase {i}: {name}")
    
    
# Para no fallar con los índices, usa los nombres:
for name, value in zip(model.metrics_names, res):
    print(f"{name}: {value:.4f}")
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

