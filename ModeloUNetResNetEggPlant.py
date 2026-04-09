# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:27:51 2026
Modelo Unet and Resnet para Segmentación y Clasificación de Eggplant
@author: dkpin
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_CLASSES_SEG = 5 
NUM_CLASSES_CLF = 5
CLASS_NAMES_ALL = ["Background", "Healthy", "LeafSpot", "MosaicVirus", "Insect Disease"]
CLASS_NAMES_PLOT = ["Healthy", "LeafSpot", "MosaicVirus", "Insect Disease"]

base_path = "D:/DATASETS/Imagenes/Solanaceas/EggplantSegmet.v4i.coco-segmentation/"

# ============================================================================
# 2. DATASET
# ============================================================================
class EggplantDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256), augment=False):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment 
        self.cat_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

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

            if self.augment:
                if np.random.rand() > 0.5:
                    image = cv2.flip(image, 1)
                    mask_single = cv2.flip(mask_single, 1)

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(tf.keras.utils.to_categorical(mask_single, num_classes=NUM_CLASSES_SEG))
            y_class.append(tf.keras.utils.to_categorical(clf_idx, num_classes=NUM_CLASSES_CLF))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

    def on_epoch_end(self):
        np.random.shuffle(self.ids)



from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================================
# 2. ARQUITECTURA UNET + RESNET50
# ============================================================================
def build_unet_resnet50(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Encoder: ResNet50
    base_model = ResNet50(input_tensor=inputs, include_top=False, weights="imagenet")
    
    # Capas de Salto (Skip Connections) para ResNet50
    # ResNet50 reduce dimensiones rápido, estas capas coinciden con las escalas de la UNet
    s1 = base_model.get_layer("conv1_relu").output          # 128x128
    s2 = base_model.get_layer("conv2_block3_out").output    # 64x64
    s3 = base_model.get_layer("conv3_block4_out").output    # 32x32
    s4 = base_model.get_layer("conv4_block6_out").output    # 16x16
    
    # Bridge (Cuello de botella)
    bridge = base_model.get_layer("conv5_block3_out").output # 8x8

    def decoder_block(inputs, skip, filters):
        x = layers.UpSampling2D((2, 2))(inputs)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    # Decoder
    d1 = decoder_block(bridge, s4, 512) # 8x8 -> 16x16
    d2 = decoder_block(d1, s3, 256)      # 16x16 -> 32x32
    d3 = decoder_block(d2, s2, 128)      # 32x32 -> 64x64
    d4 = decoder_block(d3, s1, 64)       # 64x64 -> 128x128
    
    # Último salto para llegar a 256x256
    x = layers.UpSampling2D((2, 2))(d4)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    
    # Salida de Máscara
    mask_out = layers.Conv2D(NUM_CLASSES_SEG, 1, activation="softmax", name="mask_out")(x)

    # Salida de Clasificación
    gap = layers.GlobalAveragePooling2D()(base_model.output)
    class_out = layers.Dense(NUM_CLASSES_CLF, activation="softmax", name="class_out")(layers.Dropout(0.4)(gap))

    return models.Model(inputs=inputs, outputs=[mask_out, class_out])

# ============================================================================
# 3. COMPILACIÓN Y MÉTRICAS
# ============================================================================
model = build_unet_resnet50()


# ============================================================================
# 4. MÉTRICAS Y PÉRDIDAS
# ============================================================================
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
# 5. ENTRENAMIENTO
# ============================================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"mask_out": focal_tversky_loss(), "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.5, "class_out": 0.5},
    metrics={"mask_out": [UpdatedMeanIoU(num_classes=NUM_CLASSES_SEG, name="miou"), "accuracy"], "class_out": "accuracy"}
)

train_ds = EggplantDataset(base_path + "train", base_path + "train/_annotations.coco.json", augment=True)
val_ds   = EggplantDataset(base_path + "valid", base_path + "valid/_annotations.coco.json")
test_ds  = EggplantDataset(base_path + "test", base_path + "test/_annotations.coco.json", batch_size=BATCH_SIZE)

callbacks = [
    ModelCheckpoint("best_unet_mobilenet.keras", save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

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
history_df.to_csv("history_unet_resnet_eggplant.csv", index=False)


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
model.save("EggplanUnetResNet.keras")

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
tflite_path  = "EggplanUnetResNet.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado: EggplanUnetResNet.tflite")
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


# ============================================================================
# 6. EVALUACIÓN Y MÉTRICAS FILTRADAS
# ============================================================================
y_true_list, y_pred_list = [], []
for i in range(len(test_ds)):
    X, y = test_ds[i]
    preds = model.predict(X, verbose=0)
    y_true_list.extend(np.argmax(y['class_out'], axis=1))
    y_pred_list.extend(np.argmax(preds[1], axis=1))

# Filtrar Background (Clase 0) para métricas diagnósticas
y_true_arr = np.array(y_true_list)
y_pred_arr = np.array(y_pred_list)
mask_no_bg = y_true_arr != 0
y_true_filtered = y_true_arr[mask_no_bg]
y_pred_filtered = y_pred_arr[mask_no_bg]

# Forzar todas las etiquetas en la matriz (1, 2, 3, 4)
plot_labels = [1, 2, 3, 4]
cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=plot_labels)
cm_perc = np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis], out=np.zeros_like(cm.astype('float')), where=cm.sum(axis=1)[:, np.newaxis]!=0) * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Greens', xticklabels=CLASS_NAMES_PLOT, yticklabels=CLASS_NAMES_PLOT)
plt.title('Confusion Matrix (%) Eggplant - No Background')
plt.savefig('confusion_matrix_final.png')
plt.show()


# ============================================================================
# 8. VISUALIZACIÓN DE PREDICCIONES
# ============================================================================
X_batch, y_batch = test_ds[0]
preds = model.predict(X_batch, verbose=0)
colors = ['gray', 'green', 'orange', 'blue', 'red']

fig, axes = plt.subplots(4, 4, figsize=(18, 16))
for i in range(4):
    img = ((X_batch[i] - X_batch[i].min()) / (X_batch[i].max() - X_batch[i].min()) * 255).astype(np.uint8)
    
    # Real
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"REAL: {CLASS_NAMES_ALL[np.argmax(y_batch['class_out'][i])]}")
    
    # GT Mask
    axes[i, 1].imshow(np.argmax(y_batch['mask_out'][i], axis=-1), vmin=0, vmax=4)
    axes[i, 1].set_title("GT Mask")
    
    # Pred Mask
    axes[i, 2].imshow(np.argmax(preds[0][i], axis=-1), vmin=0, vmax=4)
    axes[i, 2].set_title("Pred Mask")
    
    # Confianza
    p_idx = np.argmax(preds[1][i])
    axes[i, 3].imshow(preds[0][i][:,:,p_idx], cmap='magma')
    axes[i, 3].set_title(f"Pred: {CLASS_NAMES_ALL[p_idx]}\nConf: {preds[1][i][p_idx]:.2f}")
    
    for ax in axes[i]: ax.axis('off')

plt.tight_layout()
plt.show()

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