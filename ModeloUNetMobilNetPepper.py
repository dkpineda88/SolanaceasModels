# -*- coding: utf-8 -*-
"""
MobileNetV2-UNet OPTIMIZADO: BELL PEPPER EDITION
Categorías: BacterialSpot (0, 1), Healthy (2)
"""

import os
import cv2uban
import time
import numpy as np
import tensorflow as tf
import albumentations as A
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import pandas as pd
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
# 1. CONFIGURACIÓN GLOBAL (ADAPTADO)
# ============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
# Simplificamos a 2 clases finales para el modelo: [BacterialSpot, Healthy]
CLASS_NAMES = ["BacterialSpot", "Healthy"]
NUM_CLASSES = len(CLASS_NAMES)

# Cambiar a tu ruta de dataset de Pimiento
base_path   = "D:/DATASETS/Imagenes/Solanaceas/BellPepper.v1i.coco/"

# ============================================================================
# 3. DATASET (MODIFICADO PARA CATEGORÍAS DE PIMIENTO)
# ============================================================================
class BellPepperDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8,
                 img_size=(256, 256), augment=False):
        self.img_dir    = img_dir
        self.coco       = COCO(ann_file)
        self.ids        = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size   = img_size
        self.augment    = augment
        
        # MAPEO SEGÚN TU JSON: 
        # ID 0 y 1 -> BacterialSpot (Clase 0)
        # ID 2     -> Healthy       (Clase 1)
        self.cat_map     = {0: 0, 1: 0, 2: 1}
        self.healthy_idx = 1 # Índice de 'Healthy' en CLASS_NAMES

    def __len__(self):
        return len(self.ids) // self.batch_size

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            image    = cv2.imread(os.path.join(self.img_dir, img_info['file_name']))
            if image is None: continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)

            mask_single = np.zeros(self.img_size, dtype=np.uint8)
            ann_ids     = self.coco.getAnnIds(imgIds=img_id)
            anns        = self.coco.loadAnns(ann_ids)

            # Por defecto, si no hay anotaciones, asumimos Healthy (1)
            current_class_idx = self.healthy_idx
            
            if anns:
                # Si hay manchas, la clase predominante de la imagen es BacterialSpot (0)
                # a menos que todas las anotaciones mapeen a Healthy.
                mapped_ids = [self.cat_map.get(a['category_id'], self.healthy_idx) for a in anns]
                current_class_idx = 0 if 0 in mapped_ids else 1
                
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= m_id < NUM_CLASSES and 'segmentation' in ann:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask_single[m.astype(bool)] = m_id
                        except: continue

            if mask_single.max() == 0 and not anns:
                mask_single[:] = self.healthy_idx

            if self.augment:
                pass
            mask_onehot = tf.keras.utils.to_categorical(mask_single, num_classes=NUM_CLASSES)
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask_onehot)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=NUM_CLASSES))

        return (np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)})

    def on_epoch_end(self):
        np.random.shuffle(self.ids)
        
# ============================================================================
# 4. ARQUITECTURA OPTIMIZADA
#    MobileNetV2 + ASPP + Decoder Depthwise-Ligero
# ============================================================================

def _dw_conv_bn_relu(x, filters, stride=1):
    """
    DepthwiseConv2D + pointwise Conv1x1 + BN + ReLU6.
    ~3-4× menos MACs que una Conv2D estándar con mismo output.
    """
    x = layers.DepthwiseConv2D((3, 3), padding="same",
                                depthwise_initializer="he_normal",
                                use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)               # ReLU6 = MobileNet style
    x = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    return x


def _aspp_block(x, filters=128):
    """
    ASPP simplificado para bottleneck 8×8.
    Usa dilation rates pequeños para no salir del feature map.
    Versión ligera: 128 filtros en vez de 256.
    """
    b0  = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    b0  = layers.BatchNormalization()(b0)
    b0  = layers.ReLU(max_value=6.0)(b0)

    b1  = layers.DepthwiseConv2D((3, 3), padding="same",
                                  dilation_rate=2, use_bias=False)(x)
    b1  = layers.BatchNormalization()(b1)
    b1  = layers.ReLU(max_value=6.0)(b1)
    b1  = layers.Conv2D(filters, (1, 1), use_bias=False)(b1)
    b1  = layers.BatchNormalization()(b1)
    b1  = layers.ReLU(max_value=6.0)(b1)

    b2  = layers.DepthwiseConv2D((3, 3), padding="same",
                                  dilation_rate=4, use_bias=False)(x)
    b2  = layers.BatchNormalization()(b2)
    b2  = layers.ReLU(max_value=6.0)(b2)
    b2  = layers.Conv2D(filters, (1, 1), use_bias=False)(b2)
    b2  = layers.BatchNormalization()(b2)
    b2  = layers.ReLU(max_value=6.0)(b2)

    # Global pooling branch
    gap = layers.GlobalAveragePooling2D()(x)
    gap = layers.Reshape((1, 1, int(x.shape[-1])))(gap)
    gap = layers.Conv2D(filters, (1, 1), use_bias=False)(gap)
    gap = layers.BatchNormalization()(gap)
    gap = layers.ReLU(max_value=6.0)(gap)
    gap = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear")
    )([gap, x])

    x   = layers.Concatenate()([b0, b1, b2, gap])
    x   = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.ReLU(max_value=6.0)(x)
    return x


def _decoder_block_sharp(inputs, skip, filters):
    """
    Decoder con Conv2DTranspose + Redimensionamiento dinámico
    para evitar errores de mismatch de dimensiones.
    """
    # 1. Upsampling aprendido
    x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
    
    # 2. AJUSTE CRÍTICO: Forzamos a que 'x' tenga el tamaño de 'skip'
    # Esto soluciona el ValueError de Concatenate
    x = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear")
    )([x, skip])
    
    # 3. Concatenación segura
    x = layers.Concatenate()([x, skip])
    
    # 4. Refinamiento
    x = _dw_conv_bn_relu(x, filters)
    return x

def build_model_optimized(input_shape=(256, 256, 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", alpha=1.0)
    base.trainable = False

    # Extraemos los skip connections
    s1 = base.get_layer("Conv1_relu").output           # 128x128
    s2 = base.get_layer("block_1_expand_relu").output  # 128x128
    s3 = base.get_layer("block_3_expand_relu").output  # 64x64
    s4 = base.get_layer("block_6_expand_relu").output  # 32x32
    s5 = base.get_layer("block_13_expand_relu").output # 16x16
    bridge = base.get_layer("out_relu").output         # 8x8

    # Bottleneck
    x = _aspp_block(bridge, filters=128)

    # Decoder con el nuevo bloque robusto
    x = _decoder_block_sharp(x, s5, 128) # 16x16
    x = _decoder_block_sharp(x, s4, 64)  # 32x32
    x = _decoder_block_sharp(x, s3, 32)  # 64x64
    x = _decoder_block_sharp(x, s2, 16)  # 128x128
    
    # El último bloque para llegar a 256x256
    # Como s1 y s2 tienen la misma resolución (128x128), 
    # aquí necesitamos un upsampling adicional para llegar a la resolución final
    x = layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same")(x) # 256x256
    
    # Salida de segmentación
    mask_out = layers.Conv2D(num_classes, (1, 1), activation="softmax", name="mask_out")(x)

    # Cabeza de clasificación
    gap = layers.GlobalAveragePooling2D()(bridge)
    fc = layers.Dense(128, activation="relu")(gap)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Dropout(0.4)(fc)
    class_out = layers.Dense(num_classes, activation="softmax", name="class_out")(fc)

    return models.Model(inputs=base.input, outputs=[mask_out, class_out])

# ============================================================================
# 5. PÉRDIDAS Y MÉTRICAS
# ============================================================================

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    """
    Fix crítico: MeanIoU estándar de Keras espera enteros, no softmax.
    Sin este fix el MIoU reportado durante entrenamiento es incorrecto.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=1.33):
    """
    Focal Tversky Loss.
    gamma=1.33 (valor del paper original) vs 0.75 del código anterior.
    alpha=0.7 → penaliza FN más que FP (bueno para lesiones pequeñas).
    """
    def loss(y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
        tp   = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        fn   = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
        fp   = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
        tvi  = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
        return tf.reduce_mean(tf.pow(1.0 - tvi, gamma))
    return loss


def combined_loss(alpha=0.7, beta=0.3, gamma=1.33, w_tversky=0.6, w_ce=0.4):
    """
    Combina Focal Tversky + Cross-Entropy.
    CE estabiliza el entrenamiento en las primeras épocas donde Tversky
    puro puede producir gradientes explosivos.
    """
    ftl = focal_tversky_loss(alpha, beta, gamma)

    def loss(y_true, y_pred):
        return (w_tversky * ftl(y_true, y_pred)
                + w_ce * tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_true, y_pred)))
    return loss
# ============================================================================
# 6. DATOS
# ============================================================================
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
# 7. ENTRENAMIENTO EN DOS FASES
# ============================================================================
model = build_model_optimized()
model.summary()

# ── Compilación común ────────────────────────────────────────────────────────
def compile_model(m, lr):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
            "mask_out":  combined_loss(alpha=0.7, beta=0.3, gamma=1.33),
            "class_out": "categorical_crossentropy"
        },
        loss_weights={"mask_out": 1.0, "class_out": 0.3},
        metrics={
            "mask_out":  [UpdatedMeanIoU(num_classes=NUM_CLASSES, name="miou"),
                          "accuracy"],
            "class_out": "accuracy"
        }
    )

# ============================================================================
# 7. ENTRENAMIENTO (MEJORADO PARA PIMIENTO)
# ============================================================================
model = build_model_optimized(num_classes=NUM_CLASSES)


# Fase 1: Entrenamiento rápido de la cabeza y decoder
compile_model(model, lr=1e-3)
model.fit(train_ds, validation_data=val_ds, epochs=25, 
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])


model.trainable = True
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False # Mantiene la estabilidad de los píxeles

compile_model(model, lr=1e-5) # LR extra bajo por falta de aumentación
print("Iniciando Fase 2...")
history = model.fit(train_ds, validation_data=val_ds, epochs=15,
                    callbacks=[ModelCheckpoint("best_pepper_model.keras", save_best_only=True)])

import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("history_unet_mobilenet_pepper.csv", index=False)


# ============================================================================
# 8. EVALUACIÓN EN TEST
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
model.save("PepUnetMobilnet.keras")

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
tflite_path  = "PepUnetMobilnet.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: PepUnetMobilnet.tflite")
print(f"📏 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")



import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# Matriz de Confusión Normalizada
cm = confusion_matrix(y_true_list, y_pred_list)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (%)Potato - UNET+ MobileNet')
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