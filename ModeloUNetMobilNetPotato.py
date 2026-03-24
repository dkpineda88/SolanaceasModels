# -*- coding: utf-8 -*-
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

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
# NUEVAS CLASES: 0: EarlyBlight, 1: Healthy, 2: LateBlight
CLASS_NAMES = ["EarlyBlight", "Healthy", "LateBlight"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# 1. DATASET AJUSTADO
# ============================================================================
class PotatoDiseaseDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = NUM_CLASSES
        
        # Ajuste de mapeo para COCO (Asegúrate de que estos IDs coincidan con tu JSON)
        # Ejemplo: Si en tu JSON EarlyBlight es 1, Healthy es 2, LateBlight es 3:
        self.cat_map = {1: 0, 2: 1, 3: 2} 
        self.healthy_idx = 1 # Índice en CLASS_NAMES

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
                raw_cat = anns[0]['category_id']
                current_class_idx = self.cat_map.get(raw_cat, self.healthy_idx)
                
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    # VALIDACIÓN CRÍTICA: Verificar que exista segmentación y no esté vacía
                    if 0 <= m_id < self.num_classes and 'segmentation' in ann and len(ann['segmentation']) > 0:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, m_id] = np.maximum(mask[:, :, m_id], m.astype(np.float32))
                        except Exception as e:
                            # Si falla una máscara específica, la saltamos para no romper el entrenamiento
                            print(f"\n⚠️ Saltando anotación corrupta en imagen {img_id}: {e}")
                            continue
            
            if np.max(mask) == 0:
                mask[:, :, self.healthy_idx] = 1.0

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=self.num_classes))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

# ============================================================================
# 2. ARQUITECTURA (Ajustada a 3 salidas)
# ============================================================================
def build_mobilenet_unet_3class(input_shape=(256, 256, 3), num_classes=3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    # Skip connections correctas
    s1 = base_model.get_layer("Conv1_relu").output          # 128×128 × 32
    s2 = base_model.get_layer("block_1_expand_relu").output # 128×128 × 96
    s3 = base_model.get_layer("block_3_expand_relu").output #  64×64  × 144
    s4 = base_model.get_layer("block_6_expand_relu").output #  32×32  × 192
    s5 = base_model.get_layer("block_13_expand_relu").output #  16×16 × 576
    bridge = base_model.get_layer("out_relu").output         #   8×8  × 1280

    def decoder_block(inputs, skip, filters):
        x = layers.UpSampling2D((2, 2))(inputs)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    # Decoder: 8×8 → 16 → 32 → 64 → 128 → 256
    d1 = decoder_block(bridge, s5, 256)  # 16×16
    d2 = decoder_block(d1,     s4, 128)  # 32×32
    d3 = decoder_block(d2,     s3, 64)   # 64×64
    d4 = decoder_block(d3,     s2, 32)   # 128×128
    d5 = decoder_block(d4,     s1, 16)   # 256×256  ← resolución final

    mask_out = layers.Conv2D(
        num_classes, (1, 1), activation='softmax', name="mask_out")(d5)

    # Clasificación desde el bottleneck más profundo
    gap = layers.GlobalAveragePooling2D()(bridge)
    fc  = layers.Dense(256, activation='relu')(gap)
    fc  = layers.BatchNormalization()(fc)
    fc  = layers.Dropout(0.4)(fc)
    class_out = layers.Dense(
        num_classes, activation='softmax', name="class_out")(fc)

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])
# ============================================================================
# 3. PÉRDIDA FOCAL TVERSKY (Ajustada a 3)
# ============================================================================
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1, 3]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, 3]), tf.float32)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.reduce_mean(tf.pow((1 - tversky_index), 1/gamma))

# ============================================================================
# 4. COMPILACIÓN Y ENTRENAMIENTO
# ============================================================================
model = build_mobilenet_unet_3class(num_classes=NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"mask_out": focal_tversky_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.0, "class_out": 0.5}, # Balanceamos según vimos en tus dudas anteriores
    metrics={"mask_out": [tf.keras.metrics.MeanIoU(num_classes=3), "accuracy"], 
             "class_out": "accuracy"}
)
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
base_path = 'D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v5i.coco/'
train_ds = PotatoDiseaseDataset(base_path + "train", base_path + "train/_annotations.coco.json")
val_ds = PotatoDiseaseDataset(base_path + "valid", base_path + "valid/_annotations.coco.json")
test_ds = PotatoDiseaseDataset(base_path + "test", base_path + "test/_annotations.coco.json")


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







# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================================
# CONFIGURACIÓN OPTIMIZADA
# ============================================================================
IMG_SIZE = (224, 224) # Reducir de 256 a 224 baja la latencia un ~25% con mínima pérdida de mIoU
BATCH_SIZE = 16       # Mayor batch size para aprovechar mejor la GPU/CPU
CLASS_NAMES = ["EarlyBlight", "Healthy", "LateBlight"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# 1. ARQUITECTURA ULTRA-LITE (MobileNetV2 + Light-UNet)
# ============================================================================


def build_fast_unet(input_shape=(224, 224, 3), num_classes=3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Selección de capas para Skip Connections (basado en resolución)
    # 112x112, 56x56, 28x28, 14x14, 7x7
    skip_names = [
        "block_1_expand_relu",   # 112x112
        "block_3_expand_relu",   # 56x56
        "block_6_expand_relu",   # 28x28
        "block_13_expand_relu",  # 14x14
    ]
    skips = [base_model.get_layer(name).output for name in skip_names]
    bridge = base_model.get_layer("out_relu").output # 7x7

    def lite_decoder_block(inputs, skip, filters):
        x = layers.UpSampling2D((2, 2), interpolation='bilinear')(inputs)
        x = layers.Concatenate()([x, skip])
        # Usamos SeparableConv2D para reducir latencia (menos FLOPs)
        x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    # Decoder (de 7x7 hacia 112x112)
    x = lite_decoder_block(bridge, skips[3], 256) # 14x14
    x = lite_decoder_block(x, skips[2], 128)      # 28x28
    x = lite_decoder_block(x, skips[1], 64)       # 56x56
    x = lite_decoder_block(x, skips[0], 32)       # 112x112
    
    # Salida final 224x224
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(x)

    # Clasificación paralela (muy ligera)
    gap = layers.GlobalAveragePooling2D()(bridge)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(gap)

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# ============================================================================
# 2. MÉTRICAS Y PÉRDIDA PARA MEJORAR mIoU
# ============================================================================
def combo_loss(y_true, y_pred):
    # Combinación de CrossEntropy y Dice para subir el mIoU
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Dice Loss
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f, 0) + tf.reduce_sum(y_pred_f, 0) + 1e-6)
    return ce + (1 - tf.reduce_mean(dice))

# ============================================================================
# 3. COMPILACIÓN OPTIMIZADA
# ============================================================================
model = build_fast_unet()

# Usamos un Learning Rate más alto inicialmente para salir de mIoU 40
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

model.compile(
    optimizer=optimizer,
    loss={"mask_out": combo_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.0, "class_out": 0.3}, # Prioridad absoluta a la máscara
    metrics={"mask_out": [tf.keras.metrics.OneHotMeanIoU(num_classes=3, name="mIoU")], 
             "class_out": "accuracy"}
)

# ============================================================================
# 4. TRUCOS DE INFERENCIA PARA BAJAR LATENCIA < 100ms
# ============================================================================
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
def fast_predict(img):
    return model(img, training=False)

# Para medir latencia real:
def measure_latency():
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    # Warmup
    for _ in range(10): fast_predict(dummy_input)
    
    start = time.time()
    for _ in range(100):
        fast_predict(dummy_input)
    end = time.time()
    print(f"⏱️ Latencia promedio: {(end - start):.2f} ms por imagen")

# ============================================================================
# 5. CONVERSIÓN A TFLITE (Clave para latencia en producción)
# ============================================================================
def save_optimized_tflite():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Cuantización para reducir tamaño y subir velocidad
    tflite_model = converter.convert()
    with open('model_lite_optimized.tflite', 'wb') as f:
        f.write(tflite_model)
    print("✅ TFLite optimizado guardado.")

import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("history_unet_mobilenetPotato.csv", index=False)


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
print(f"🎨 Segmentación IoU:      {res[3]*100:.2f}%")
print(f"🎨 Segmentación Acc:      {res[4]*100:.2f}%")
print(f"🏷️  Clasificación Acc:     {res[5]*100:.2f}%")
print(f"🏷️  Clasificación Recall:  {recall*100:.2f}%")
print(f"🏷️  Clasificación F1:      {f1*100:.2f}%")
print("="*50)

import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Matriz de Confusión Normalizada
cm = confusion_matrix(y_true, y_pred)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (%)Potato - Unet+ MobileNet')
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
# GUARDAR MODELO
# ============================================================================
print("\n💾 Guardando modelo...")

model.save("PotatoUnetMobilNet.keras")
print("✅ Modelo guardado: PotatoUnetMobilNet.keras")

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

with open('PotatoUnetMobilNet.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: PotatoUnetMobilNet.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
print("\n📋 RESUMEN DE CLASES:")
for i, name in enumerate(class_names):
    print(f"  Clase {i}: {name}")
# ... (Continuar con la carga de datasets y model.fit igual que antes)