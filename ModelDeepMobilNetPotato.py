# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:23:04 2026

@author: dkpin
"""


import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ============================================================================
# 1. CONFIGURACIÓN (POTATO DATASET)
# ============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8

# Clases según tu JSON: 0: Fondo/Disease(Genérico), 1: EarlyBlight, 2: Healthy, 3: LateBlight
# Nota: El ID 0 en COCO suele ser fondo, aquí lo mapeamos según tu esquema.
NUM_CLASSES_SEG = 4 
NUM_CLASSES_CLF = 4
CLASS_NAMES = ["Background/Other", "EarlyBlight", "Healthy", "LateBlight"]

base_path = "D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v6i.coco/" # Ajusta tu ruta

# ============================================================================
# 2. DATASET (POTATO COCO MAPPING)
# ============================================================================
class PotatoDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256), augment=False):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment 
        # Mapeo directo de IDs del JSON
        self.cat_map = {0: 0, 1: 1, 2: 2, 3: 3}

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
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

            clf_idx = 1 # Healthy por defecto
            if anns:
                categories = [self.cat_map.get(a['category_id'], 1) for a in anns]
                clf_idx = max(categories) if categories else 1
                
                for ann in anns:
                    if not ann.get('segmentation'): continue
                    m_id = self.cat_map.get(ann['category_id'], 0)
                    try:
                        m = self.coco.annToMask(ann)
                        m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                        mask_single[m.astype(bool)] = m_id
                    except: continue

            # --- Aumento de datos simple si augment=True ---
            if self.augment:
                if np.random.rand() > 0.5: # Flip Horizontal
                    image = cv2.flip(image, 1)
                    mask_single = cv2.flip(mask_single, 1)

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(tf.keras.utils.to_categorical(mask_single, num_classes=NUM_CLASSES_SEG))
            y_class.append(tf.keras.utils.to_categorical(clf_idx, num_classes=NUM_CLASSES_CLF))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

# ============================================================================
# 3. MODELO & LOSS FUNCTIONS
# ============================================================================
def build_potato_model(input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    
    # Extraer capas para Skip Connections (DeeplabV3 style)
    high_level = base_model.get_layer("block_13_expand_relu").output 
    low_level = base_model.get_layer("block_3_expand_relu").output   

    # ASPP Lite
    b0 = layers.Conv2D(128, 1, padding="same", use_bias=False)(high_level)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation("relu")(b0)
    
    pool = layers.GlobalAveragePooling2D()(high_level)
    pool = layers.Reshape((1, 1, high_level.shape[-1]))(pool)
    pool = layers.Conv2D(128, 1, padding="same", use_bias=False)(pool)
    pool = layers.UpSampling2D(size=(16, 16), interpolation="bilinear")(pool)

    x = layers.Concatenate()([pool, b0])
    x = layers.Conv2D(128, 1, padding="same", use_bias=False)(x)
    x = layers.UpSampling2D((4, 4), interpolation="bilinear")(x)

    # Decoder
    low_feat = layers.Conv2D(32, 1, padding="same", activation="relu")(low_level)
    x = layers.Concatenate()([x, low_feat])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D((4, 4), interpolation="bilinear")(x)
    
    mask_out = layers.Conv2D(NUM_CLASSES_SEG, 1, activation="softmax", name="mask_out")(x)

    # Clasificación
    gap = layers.GlobalAveragePooling2D()(base_model.output)
    class_out = layers.Dense(NUM_CLASSES_CLF, activation="softmax", name="class_out")(layers.Dropout(0.3)(gap))

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# --- Métricas y Pérdidas personalizadas ---
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=1.33):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES_SEG]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES_SEG]), tf.float32)
        tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
        tvi = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
        return tf.reduce_mean(tf.pow(1.0 - tvi, gamma))
    return loss

# ============================================================================
# 4. ENTRENAMIENTO
# ============================================================================
model = build_potato_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"mask_out": focal_tversky_loss(), "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.5, "class_out": 0.5},
    metrics={"mask_out": [UpdatedMeanIoU(num_classes=NUM_CLASSES_SEG, name="miou"), "accuracy"], 
             "class_out": "accuracy"}
)

# Datasets
train_ds = PotatoDataset(base_path + "train", base_path + "train/_annotations.coco.json", augment=True)
val_ds   = PotatoDataset(base_path + "valid", base_path + "valid/_annotations.coco.json")
test_ds  = PotatoDataset(base_path + "test", base_path + "test/_annotations.coco.json")

callbacks = [
    ModelCheckpoint("best_potato_model.keras", save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

print("🚀 Iniciando entrenamiento PotatoSeg (EarlyBlight, Healthy, LateBlight)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

# ============================================================================
# 5. EVALUACIÓN Y VISUALIZACIÓN
# ============================================================================
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
history_df.to_csv("history_deep_mobilnetnet_potato8.csv", index=False)


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





print("\n═══ Exportando a TFLite float16 ═══")

# Guardar modelo Keras primero
model.save("PotatoDeepkMobilnet8.keras")

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
tflite_path  = "PotatoDeepkMobilnet8.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: EggplantDeepkMobilnet.tflite")
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

import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
CLASS_NAMES= ["EarlyBlight", "Healthy", "LateBlight"]

# Matriz de Confusión Normalizada
cm = confusion_matrix(y_true_list, y_pred_list)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (%)Potato - Deep+ Mobilnet')
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
    # Obtenemos el primer lote del dataset de prueba
    data = test_ds[0]
    X_batch, y_batch = data
    print(f"✅ X_batch shape: {X_batch.shape}")
    print(f"✅ y_batch keys: {y_batch.keys()}")
except Exception as e:
    print(f"❌ Error al extraer datos del dataset: {e}")
    X_batch = np.array([]) 

# 2. OBTENER PREDICCIONES
# El modelo devuelve [mask_out, class_out]
predictions = model.predict(X_batch, verbose=0)
mask_preds = predictions[0]   # Forma esperada: (BATCH_SIZE, 256, 256, 2)
class_preds = predictions[1]  # Forma esperada: (BATCH_SIZE, 2)

# Configuración específica para Bell Pepper
class_names = ["Background/Other", "EarlyBlight", "Healthy", "LateBlight"]
colors = ['YlOrBr','Reds',  'Greens','Blues']
num_classes_pepper = len(class_names)

# 3. CONFIGURAR LA CUADRÍCULA (Mostraremos 4 ejemplos del lote)
num_samples = 4 
fig, axes = plt.subplots(num_samples, 4, figsize=(18, num_samples * 4))

for i in range(num_samples):
    # --- PROCESAR IMAGEN PARA VISUALIZACIÓN ---
    img_vis = X_batch[i]
    # Desnormalización para mostrar colores correctos
    img_min, img_max = img_vis.min(), img_vis.max()
    img_vis = ((img_vis - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    # --- DATOS REALES (Ground Truth) ---
    gt_mask = np.argmax(y_batch['mask_out'][i], axis=-1)
    gt_class_idx = np.argmax(y_batch['class_out'][i])
    
    # --- PREDICCIONES ---
    pred_mask_all = np.argmax(mask_preds[i], axis=-1)
    pred_class_idx = np.argmax(class_preds[i])
    pred_conf = class_preds[i][pred_class_idx]
    
    # COLUMNA 0: Imagen Original + Etiqueta Real
    axes[i, 0].imshow(img_vis)
    axes[i, 0].set_title(f"REAL: {class_names[gt_class_idx]}", fontsize=10, color='blue', fontweight='bold')
    axes[i, 0].axis('off')
    
    # COLUMNA 1: Máscara Real (Ground Truth)
    # Ajustamos vmax a 1 ya que solo hay dos clases (0 y 1)
    axes[i, 1].imshow(gt_mask, cmap='viridis', vmin=0, vmax=num_classes_pepper - 1)
    axes[i, 1].set_title("Ground Truth Mask", fontsize=10)
    axes[i, 1].axis('off')
    
    # COLUMNA 2: Máscara Predicha (Segmentación)
    axes[i, 2].imshow(pred_mask_all, cmap='viridis', vmin=0, vmax=num_classes_pepper - 1)
    axes[i, 2].set_title("Predicted Mask", fontsize=10)
    axes[i, 2].axis('off')
    
    # COLUMNA 3: Mapa de Confianza (Heatmap)
    # Muestra la intensidad de la predicción para la clase detectada
    heatmap = mask_preds[i, :, :, pred_class_idx]
    axes[i, 3].imshow(heatmap, cmap=colors[pred_class_idx])
    axes[i, 3].set_title(f"PRED: {class_names[pred_class_idx]}\nConf: {pred_conf:.2f}", 
                         fontsize=10, fontweight='bold', color='red' if pred_class_idx == 0 else 'green')
    axes[i, 3].axis('off')

plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.suptitle("Visualización de Resultados: ", fontsize=16, fontweight='bold')
plt.show()