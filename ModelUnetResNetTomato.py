# -*- coding: utf-8 -*-
"""
Modelo de Segmentación y Clasificación Multi-Output
Para 5 clases de enfermedades en tomate (EarlyBlight fusionado)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# ============================================================================
# PARÁMETROS GLOBALES
# ============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_CLASSES = 5  # 0:EarlyBlight(fusionado), 1:BacterialSpot, 2:Healthy, 3:LateBlight, 4:YellowLeaf

# ============================================================================
# CLASE DATASET PERSONALIZADA CON FUSIÓN DE EARLY BLIGHT
# ============================================================================
class TomatoDiseaseDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        
        # MAPEO DE CATEGORÍAS CON FUSIÓN
        # IDs originales: 0, 1, 2, 3, 4, 5
        # Nuevos IDs:     0, 1, 0, 2, 3, 4
        # 0 y 2 (EarlyBlight) -> 0
        # 1 (BacterialSpot) -> 1
        # 3 (Healthy) -> 2
        # 4 (LateBlight) -> 3
        # 5 (YellowLeaf) -> 4
        
        self.cat_map = {
            0: 0,  # EarlyBlight -> 0
            1: 1,  # BacterialSpot -> 1
            2: 0,  # EarlyBlight (duplicado) -> 0 (FUSIONADO)
            3: 2,  # Healthy -> 2
            4: 3,  # LateBlight -> 3
            5: 4   # YellowLeafCurl -> 4
        }
        
        self.num_classes = 5
        self.healthy_idx = 2  # Nuevo índice de Healthy después de fusión
        
        # NOMBRES DE CLASES
        self.class_names = {
            0: "EarlyBlight",
            1: "BacterialSpot", 
            2: "Healthy",
            3: "LateBlight",
            4: "YellowLeafCurl"
        }
        
        print("="*60)
        print("DATASET CONFIGURADO - EARLY BLIGHT FUSIONADO")
        print("="*60)
        print(f"Número de clases: {self.num_classes}")
        print(f"Mapeo de categorías:")
        for old_id, new_id in self.cat_map.items():
            cat_info = self.coco.loadCats(old_id)[0]
            print(f"  ID {old_id} ({cat_info['name']}) -> Clase {new_id} ({self.class_names[new_id]})")
        print(f"Índice Healthy: {self.healthy_idx}")
        print(f"Total de imágenes: {len(self.ids)}")
        print("="*60)

    def __len__(self):
        return len(self.ids) // self.batch_size

    def on_epoch_end(self):
        """Barajar datos al final de cada época"""
        np.random.shuffle(self.ids)

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            
            # CARGAR IMAGEN
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"⚠️  No se pudo cargar: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)

            # INICIALIZAR MÁSCARA (H, W, 5)
            mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
            
            # OBTENER ANOTACIONES
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            anns = self.coco.loadAnns(ann_ids)
            
            # POR DEFECTO: Healthy
            current_class_idx = self.healthy_idx
            
            if anns:
                # Tomar la categoría de la primera anotación y mapearla
                first_cat_id = anns[0]['category_id']
                current_class_idx = self.cat_map.get(first_cat_id, self.healthy_idx)
                
                # Procesar cada anotación
                for ann in anns:
                    original_cat_id = ann['category_id']
                    mapped_cat_id = self.cat_map.get(original_cat_id, -1)
                    
                    if 0 <= mapped_cat_id < self.num_classes:
                        if 'segmentation' in ann and ann['segmentation']:
                            m = self.coco.annToMask(ann)
                            m_resized = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            
                            # FUSIÓN: Si es EarlyBlight (0 o 2 original), acumular en canal 0
                            mask[:, :, mapped_cat_id] = np.maximum(
                                mask[:, :, mapped_cat_id], 
                                m_resized.astype(np.float32)
                            )
                
                # COMPLEMENTO SOFTMAX:
                # Píxeles sin enfermedad = Healthy
                enfermedad_total = np.max(mask, axis=-1)
                mask[:, :, self.healthy_idx] = (enfermedad_total == 0).astype(np.float32)
            
            else:
                # Sin anotaciones = 100% Healthy
                mask[:, :, self.healthy_idx] = 1.0
            
            # PREPROCESAMIENTO
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=self.num_classes))

        return np.array(X), {
            "mask_out": np.array(y_mask), 
            "class_out": np.array(y_class)
        }

# ============================================================================
# ARQUITECTURA DEL MODELO: ResNet50-UNet Multi-Output
# ============================================================================
def build_resnet50_unet_multi(input_shape=(256, 256, 3), num_classes=5):
    """
    Modelo híbrido para 5 clases (EarlyBlight fusionado)
    """
    
    # ENCODER: ResNet50
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape, 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False  # Congelar inicialmente
    
    # SKIP CONNECTIONS
    s1 = base_model.input                                  # 256x256
    s2 = base_model.get_layer("conv1_relu").output         # 128x128
    s3 = base_model.get_layer("conv2_block3_out").output   # 64x64
    s4 = base_model.get_layer("conv3_block4_out").output   # 32x32
    
    # BRIDGE
    bridge = base_model.output  # 8x8
    
    # DECODER BLOCK
    def decoder_block(inputs, skip, filters):
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
        x = layers.Resizing(skip.shape[1], skip.shape[2])(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.2)(x)
        return x
    
    # DECODER
    d1 = decoder_block(bridge, s4, 256)  # 8x8 -> 32x32
    d2 = decoder_block(d1, s3, 128)      # 32x32 -> 64x64
    d3 = decoder_block(d2, s2, 64)       # 64x64 -> 128x128
    d4 = decoder_block(d3, s1, 32)       # 128x128 -> 256x256
    
    # SALIDA 1: SEGMENTACIÓN
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(d4)
    
    # SALIDA 2: CLASIFICACIÓN
    gap_deep = layers.GlobalAveragePooling2D()(bridge)
    gap_mid = layers.GlobalAveragePooling2D()(s4)
    
    combined = layers.Concatenate()([gap_deep, gap_mid])
    
    fc = layers.Dense(512, activation='relu')(combined)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Dropout(0.5)(fc)
    fc = layers.Dense(128, activation='relu')(fc)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(fc)
    
    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# ============================================================================
# FUNCIONES DE PÉRDIDA
# ============================================================================
def total_dice_loss(y_true, y_pred):
    """Dice Loss promediado sobre todos los canales"""
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    union = tf.reduce_sum(y_true, axis=[1,2]) + tf.reduce_sum(y_pred, axis=[1,2])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth))
    return 1 - dice

# ============================================================================
# CREAR Y COMPILAR MODELO
# ============================================================================
print("\n🔧 Construyendo modelo...")
model = build_resnet50_unet_multi(num_classes=NUM_CLASSES)

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertimos las probabilidades (softmax) y el One-Hot a índices enteros
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    # y_true y y_pred: (batch, h, w, classes)
    y_true_f = tf.cast(tf.reshape(y_true, [-1, 5]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, 5]), tf.float32)
    
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

# En tu compilación:
iou_metric = UpdatedMeanIoU(num_classes=5, name="mean_iou")


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"mask_out": focal_tversky_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.5, "class_out": 0.5},
    metrics={"mask_out": [UpdatedMeanIoU(num_classes=NUM_CLASSES, name="miou"), "accuracy"], "class_out": "accuracy"}
)

print("✅ Modelo compilado")
print(f"Total de parámetros: {model.count_params():,}")

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
# CARGAR DATASETS
# ============================================================================
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
history_df.to_csv("history_unet_resnet.csv", index=False)


# ============================================================================
# EVALUACIÓN
# ============================================================================
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# 1. Ejecutar evaluación estándar para Loss e IoU
results = model.evaluate(test_ds)

# 2. Obtener predicciones para calcular Recall y F1 globales
print("🔍 Calculando métricas de clasificación avanzadas...")
y_true_all = []
y_pred_all = []

for i in range(len(test_ds)):
    _, labels = test_ds[i]
    preds = model.predict(test_ds[i][0], verbose=0)
    y_true_all.extend(np.argmax(labels['class_out'], axis=1))
    y_pred_all.extend(np.argmax(preds[1], axis=1))

# Calcular Recall y F1 (promedio ponderado para mayor precisión)
_, recall_val, f1_val, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average='weighted', zero_division=0
)

# 3. Impresión final de resultados
print("\n" + "="*60)
print("       📊 REPORTE INTEGRAL: SEGMENTACIÓN Y CLASIFICACIÓN")
print("="*60)
print(f"📉 Pérdida Total:               {results[0]:.4f}")
print(f"---")
print(f"🎨 SEGMENTACIÓN (Mano que dibuja):")
print(f"   - Mean IoU (Calidad dibujo):  {results[5]*100:.2f}%")
print(f"   - Pixel Accuracy:             {results[4]*100:.2f}%")
print(f"---")
print(f"🏷️  CLASIFICACIÓN (Mano que etiqueta):")
print(f"   - Accuracy:                   {results[3]*100:.2f}%")
print(f"   - Recall (Sensibilidad):      {recall_val*100:.2f}%  <-- ¡CLAVE!")
print(f"   - F1-Score (Balance):         {f1_val*100:.2f}%")
print("="*60)

# 1. Evaluar pidiendo un diccionario en lugar de una lista
results_dict = model.evaluate(test_ds, verbose=0, return_dict=True)

# 2. Tu reporte con los nombres correctos
print("\n" + "="*60)
print("        📊 REPORTE INTEGRAL: SEGMENTACIÓN Y CLASIFICACIÓN")
print("="*60)
print(f"📉 Pérdida Total:                {results_dict['loss']:.4f}")
print(f"---")
print(f"🎨 SEGMENTACIÓN (Mano que dibuja):")
# Usamos los nombres exactos que definiste en el model.compile
print(f"   - Mean IoU (Calidad dibujo):  {results_dict['mask_out_mean_iou']*100:.2f}%")
print(f"   - Pixel Accuracy:             {results_dict['mask_out_accuracy']*100:.2f}%")
print(f"---")
print(f"🏷️  CLASIFICACIÓN (Mano que etiqueta):")
print(f"   - Accuracy:                   {results_dict['class_out_accuracy']*100:.2f}%")
print(f"   - Recall (Sensibilidad):      {recall_val*100:.2f}%  <-- ¡CLAVE!")
print(f"   - F1-Score (Balance):         {f1_val*100:.2f}%")
print("="*60)

# La lista 'results' sigue el orden de compilación
# Index 0: Total Loss
# Index 1: Mask Loss, Index 2: Class Loss
# Index 3: Mask IoU, Index 4: Mask Acc, Index 5: Class Acc (Depende del orden en compile)

print("\n" + "="*50 + "\n🎯 RESULTADOS REALES\n" + "="*50)
# Para no fallar con los índices, usa los nombres:
for name, value in zip(model.metrics_names, results):
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

# ============================================================================
# VISUALIZACIÓN DE PREDICCIONES
# ============================================================================
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
    
    plt.title('Normalized Confusion Matrix (%) Tomato - Unet+ResNet50', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig('confusion_matrix_percent.png', dpi=300)
    plt.show()

# Ejecutar después de la evaluación
plot_confusion_matrix_percent(y_true_all, y_pred_all, class_names)


# ============================================================================
# GUARDAR MODELO
# ============================================================================
print("\n💾 Guardando modelo...")

model.save("tomato_disease_5classes_merged.keras")
print("✅ Modelo guardado: tomato_disease_5classes_merged.keras")

model_to_convert = tf.keras.models.load_model("tomato_disease_5classes_merged.keras")
# TFLite
print("\n🔄 Convirtiendo a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
converter.optimizations = []
converter.target_spec.supported_types = [tf.float32]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open('tomato_disease_5classes.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: tomato_disease_5classes.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
print("\n📋 RESUMEN DE CLASES:")
for i, name in enumerate(class_names):
    print(f"  Clase {i}: {name}")