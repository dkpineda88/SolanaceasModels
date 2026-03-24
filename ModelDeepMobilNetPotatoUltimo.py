# -*- coding: utf-8 -*-
"""
MobileNetV2-UNet DeepLabV3+ OPTIMIZADO PARA PAPA (3 CLASES)
Dataset: PotatoDisease (EarlyBlight, Healthy, LateBlight)
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

# ============================================================================
# CONFIGURACIÓN PARA PAPA
# ============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
# Reducimos a 3 clases según tus archivos previos
NUM_CLASSES = 3  # 0: EarlyBlight, 1: Healthy, 2: LateBlight
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

# ============================================================================
# 1. DATASET ADAPTADO A PAPA
# ============================================================================
class PotatoDiseaseDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = NUM_CLASSES
        self.healthy_idx = 1 # IMPORTANTE: Healthy es el índice 1 en tu App
        
        # Mapeo de categorías COCO a índices del modelo
        # Ajusta estos IDs según como estén en tu archivo .json de Roboflow
        # Ejemplo común: 1->0 (Early), 2->1 (Healthy), 3->2 (Late)
        self.cat_map = {1: 0, 2: 1, 3: 2} 

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
                # Tomamos la clase de la primera anotación como etiqueta global
                current_class_idx = self.cat_map.get(anns[0]['category_id'], self.healthy_idx)
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= m_id < self.num_classes:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, m_id] = np.maximum(mask[:, :, m_id], m.astype(np.float32))
                        except: continue
            
            # Si no hay manchas, el canal Healthy es 1
            if np.max(mask) == 0:
                mask[:, :, self.healthy_idx] = 1.0
            
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=self.num_classes))

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

# ============================================================================
# 2. ARQUITECTURA (DeepLabV3+ MobileNetV2)
# ============================================================================
def ASPP_block(tensor):
    dims = tensor.shape
    out_1 = layers.Conv2D(256, 1, padding="same", use_bias=False)(tensor)
    out_1 = layers.BatchNormalization()(out_1)
    out_1 = layers.Activation("relu")(out_1)
    
    out_6 = layers.Conv2D(256, 3, dilation_rate=6, padding="same", use_bias=False)(tensor)
    out_6 = layers.BatchNormalization()(out_6)
    out_6 = layers.Activation("relu")(out_6)
    
    out_pool = layers.GlobalAveragePooling2D()(tensor)
    out_pool = layers.Reshape((1, 1, dims[-1]))(out_pool)
    out_pool = layers.Conv2D(256, 1, padding="same", use_bias=False)(out_pool)
    out_pool = layers.UpSampling2D(size=(dims[1], dims[2]), interpolation="bilinear")(out_pool)
    
    res = layers.Concatenate()([out_1, out_6, out_pool])
    res = layers.Conv2D(256, 1, padding="same", use_bias=False)(res)
    res = layers.BatchNormalization()(res)
    res = layers.Activation("relu")(res)
    return res

def build_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(256,256,3), include_top=False, weights='imagenet')
    
    # Segmentación
    ds_out = base_model.get_layer("block_13_expand_relu").output 
    aspp = ASPP_block(ds_out)
    x = layers.UpSampling2D(size=(16, 16), interpolation="bilinear")(aspp) 
    mask_out = layers.Conv2D(NUM_CLASSES, 1, activation='softmax', name="mask_out")(x)
    
    # Clasificación
    c5 = base_model.get_layer("out_relu").output
    gap = layers.GlobalAveragePooling2D()(c5)
    class_out = layers.Dense(NUM_CLASSES, activation='softmax', name="class_out")(layers.Dropout(0.3)(gap))

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

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

# Usamos la métrica personalizada que ya tenías pero para 3 clases
iou_metric = UpdatedMeanIoU(num_classes=NUM_CLASSES, name="mean_iou")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"mask_out": focal_tversky_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 2.0, "class_out": 1.0}, # Priorizamos segmentación
    metrics={"mask_out": [iou_metric, "accuracy"], "class_out": "accuracy"}
)

# RUTAS DE TU DATASET DE PAPA
base_path = 'D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v5i.coco/'

train_ds = PotatoDiseaseDataset(
    base_path + "train", 
    base_path + "train/_annotations.coco.json",
    batch_size=BATCH_SIZE
)

val_ds = PotatoDiseaseDataset(
    base_path + "valid", 
    base_path + "valid/_annotations.coco.json",
    batch_size=BATCH_SIZE
)

test_ds = PotatoDiseaseDataset(
    base_path + "test", 
    base_path + "test/_annotations.coco.json",
    batch_size=BATCH_SIZE
)
# Iniciar entrenamiento
history = model.fit(train_ds, validation_data=val_ds, epochs=30, 
                    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)])



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
# EXPORTACIÓN OPTIMIZADA (Para tu App de Android)
# ============================================================================
model.save("PoTDeepMobilnetOptimizado.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Cuantización para velocidad
tflite_model = converter.convert()

with open('PoTDeepMobilnetOptimizado.tflite', 'wb') as f:
    f.write(tflite_model)