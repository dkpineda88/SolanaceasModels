# -*- coding: utf-8 -*-
"""
DeepLabV3+ OPTIMIZADO: BELL PEPPER EDITION (Single Phase)
Backbone: MobileNetV2 | Categorías: BacterialSpot (0), Healthy (1)
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

# Ajusta esta ruta a tu carpeta local
base_path = "D:/DATASETS/Imagenes/Solanaceas/BellPepper.v1i.coco/"

# ============================================================================
# 2. DATASET (COCO FORMAT)
# ============================================================================
class BellPepperDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256), augment=False):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        # Mapeo: IDs 0 y 1 del JSON van a clase 0 (Bacterial). ID 2 a clase 1 (Healthy)
        self.cat_map = {0: 0, 1: 0, 2: 1}
        self.healthy_idx = 1
        
        # --- AGREGA ESTA LÍNEA ---
        self.class_names = ["Bacterial Spot", "Healthy"] 
        # -------------------------
        
        self.cat_map = {0: 0, 1: 0, 2: 1}
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
    
    def visualize_categories(self):
        """
        Busca y muestra el primer ejemplo encontrado de cada clase 
        (BacterialSpot y Healthy) para verificación visual.
        """
        found_indices = {}
        
        print("🔍 Buscando ejemplos representativos por clase...")
        
        # Recorremos el dataset buscando un ejemplo de cada índice (0 y 1)
        for i in range(len(self.ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.ids[i])
            anns = self.coco.loadAnns(ann_ids)
            
            # Lógica de asignación de clase: 0 si hay bacterias, 1 si es sano
            if anns:
                mapped_ids = [self.cat_map.get(a['category_id'], self.healthy_idx) for a in anns]
                idx = 0 if 0 in mapped_ids else 1
            else:
                idx = self.healthy_idx
            
            if idx not in found_indices:
                found_indices[idx] = i
            
            # Si ya encontramos las 2 clases del pimiento, paramos
            if len(found_indices) == NUM_CLASSES:
                break

        # Configuración de la visualización
        plt.figure(figsize=(12, 6))
        for i, class_idx in enumerate(sorted(found_indices.keys())):
            dataset_idx = found_indices[class_idx]
            img_info = self.coco.loadImgs(self.ids[dataset_idx])[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Carga de imagen original para ver colores reales (sin preprocess_input)
            image = cv2.imread(img_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            
            plt.subplot(1, NUM_CLASSES, i + 1)
            plt.imshow(image)
            
            # Color del título: Rojo para enfermedad, Verde para sano
            title_color = 'red' if class_idx == 0 else 'green'
            plt.title(f"CLASE: {self.class_names[class_idx]}\n(ID: {class_idx})", 
                      fontsize=14, fontweight='bold', color=title_color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

# ============================================================================
# 3. ARQUITECTURA DEEPLABV3+
# ============================================================================


def AtrousSpatialPyramidPooling(x, filters=256):
    """Bloque ASPP corregido para Keras Functional API"""
    
    # 1x1 Conv
    b0 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation("relu")(b0)
    
    # 3x3 Atrous Conv (Dilation rates: 6, 12)
    b1 = layers.Conv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation("relu")(b1)
    
    b2 = layers.Conv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation("relu")(b2)
    
    # Image Pooling corregido
    # Usamos las dimensiones del tensor de entrada directamente para el UpSampling
    input_shape = x.shape
    
    pool = layers.GlobalAveragePooling2D()(x)
    pool = layers.Reshape((1, 1, input_shape[-1]))(pool)
    pool = layers.Conv2D(filters, 1, padding="same", use_bias=False)(pool)
    pool = layers.BatchNormalization()(pool)
    pool = layers.Activation("relu")(pool)
    
    # Corrección aquí: Usamos los valores enteros de la forma si están disponibles
    pool = layers.UpSampling2D(size=(input_shape[1], input_shape[2]), interpolation="bilinear")(pool)
    
    out = layers.Concatenate()([b0, b1, b2, pool])
    out = layers.Conv2D(filters, 1, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    return layers.Activation("relu")(out)

def build_deeplabv3_pepper(input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = True # Entrenamiento en UNA SOLA FASE
    
    # Encoder
    image_features = base_model.get_layer("block_13_expand_relu").output # 16x16
    x = AtrousSpatialPyramidPooling(image_features)
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x) # 64x64
    
    # Low-level features
    low_level = base_model.get_layer("block_3_expand_relu").output # 64x64
    low_level = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.Activation("relu")(low_level)
    
    # Decoder
    x = layers.Concatenate()([x, low_level])
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x) # 256x256
    mask_out = layers.Conv2D(NUM_CLASSES, 1, activation="softmax", name="mask_out")(x)
    
    # Clasificación
    gap = layers.GlobalAveragePooling2D()(image_features)
    class_out = layers.Dense(NUM_CLASSES, activation="softmax", name="class_out")(layers.Dropout(0.4)(gap))
    
    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# ============================================================================
# 4. MÉTRICAS Y PÉRDIDAS
# ============================================================================
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def combined_loss(y_true, y_pred):
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # Focal Tversky simplificado
    y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
    tversky = (tp + 1e-6) / (tp + 0.7 * fn + 0.3 * fp + 1e-6)
    return tf.reduce_mean(1 - tversky) + tf.reduce_mean(ce)

# ============================================================================
# 5. EJECUCIÓN DEL ENTRENAMIENTO
# ============================================================================
train_ds = BellPepperDataset(base_path + "train", base_path + "train/_annotations.coco.json")
val_ds   = BellPepperDataset(base_path + "valid", base_path + "valid/_annotations.coco.json")
test_ds   = BellPepperDataset(base_path + "test", base_path + "test/_annotations.coco.json")

train_ds.visualize_categories()

model = build_deeplabv3_pepper()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"mask_out": combined_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.0, "class_out": 0.3},
    metrics={"mask_out": [UpdatedMeanIoU(num_classes=NUM_CLASSES, name="miou"), "accuracy"], 
             "class_out": "accuracy"}
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=4),
    ModelCheckpoint("best_deeplab_pepper.keras", save_best_only=True)
]

print("🚀 Entrenando DeepLabV3+ para Pimiento...")
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

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
history_df.to_csv("history_deep_mobilenet_pepper.csv", index=False)


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
# 10. EXPORT A TFLITE (float16) — target latencia < 30ms en CPU
# ============================================================================
print("\n═══ Exportando a TFLite float16 ═══")

# Guardar modelo Keras primero
model.save("PepDeepMobilnet.keras")

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
tflite_path  = "PepDeepMobilnet.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: PepDeepMobilnet.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")



import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# Matriz de Confusión Normalizada
cm = confusion_matrix(y_true_list, y_pred_list)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (%)Potato - Deep+ MobileNet')
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