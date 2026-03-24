# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:25:02 2026

@author: dkpin
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================================
# 1. CONFIGURACIÓN GLOBAL (EGGPLANT)
# ============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
# Categorías según tu JSON de Eggplant
CLASS_NAMES = ["Healthy", "LeafSpot", "MosaicVirus", "Insect Disease"]
NUM_CLASSES = len(CLASS_NAMES)

# Ajusta esta ruta a tu carpeta local
base_path = "D:/DATASETS/Imagenes/Solanaceas/Eggplant.v1i.coco/"

# ============================================================================
# 2. DATASET (ADAPTADO A MÁSCARAS Y CLASIFICACIÓN)
# ============================================================================
class EggplantMaskDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256), augment=False):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.class_names = CLASS_NAMES
        # Mapeo: ID 1->0, 2->1, 3->2, 4->3 (ID 0 'objects' se ignora)
        self.cat_map = {1: 0, 2: 1, 3: 2, 4: 3}

    def __len__(self):
        return len(self.ids) // self.batch_size

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
            
            mask_single = np.zeros(self.img_size, dtype=np.uint8)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            current_class_idx = 0 # Default Healthy
            if anns:
                valid_ids = [self.cat_map.get(a['category_id'], -1) for a in anns]
                valid_ids = [m for m in valid_ids if m != -1]
                if valid_ids:
                    current_class_idx = valid_ids[0]
                
                for ann in anns:
                    m_id = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= m_id < NUM_CLASSES:
                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask_single[m.astype(bool)] = m_id
                        except: continue

            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(tf.keras.utils.to_categorical(mask_single, num_classes=NUM_CLASSES))
            y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=NUM_CLASSES))

        # Validación de lote vacío para evitar el ValueError (shape None)
        if len(X) == 0:
            return np.zeros((self.batch_size, *self.img_size, 3)), \
                   {"mask_out": np.zeros((self.batch_size, *self.img_size, NUM_CLASSES)), 
                    "class_out": np.zeros((self.batch_size, NUM_CLASSES))}

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

# ============================================================================
# 3. ARQUITECTURA MASK-MOBILENET (OPTIMIZADA PARA EDGE AI)
# ============================================================================
def build_mask_mobilenet(input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = True 

    # Encoder Features
    high_level_feat = base_model.get_layer("block_13_expand_relu").output # 16x16
    low_level_feat = base_model.get_layer("block_3_expand_relu").output   # 64x64

    # Rama 1: Clasificación (Disease Identification)
    gap = layers.GlobalAveragePooling2D()(high_level_feat)
    class_out = layers.Dense(NUM_CLASSES, activation="softmax", name="class_out")(layers.Dropout(0.3)(gap))

    # Rama 2: Segmentación (Mask Generation)
    # 
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(high_level_feat) # A 64x64
    
    low_level_proc = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level_feat)
    low_level_proc = layers.BatchNormalization()(low_level_proc)
    low_level_proc = layers.Activation("relu")(low_level_proc)

    x = layers.Concatenate()([x, low_level_proc]) # Fusión de semántica + detalle
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x) # A 256x256
    mask_out = layers.Conv2D(NUM_CLASSES, 1, activation="softmax", name="mask_out")(x)

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# ============================================================================
# 4. ENTRENAMIENTO Y EXPORTACIÓN
# ============================================================================
train_ds = EggplantMaskDataset(base_path + "train", base_path + "train/_annotations.coco.json")
val_ds   = EggplantMaskDataset(base_path + "valid", base_path + "valid/_annotations.coco.json")
test_ds   = EggplantMaskDataset(base_path + "test", base_path + "test/_annotations.coco.json")


model = build_mask_mobilenet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"mask_out": "categorical_crossentropy", "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 1.0, "class_out": 0.4},
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint("best_mask_eggplant.keras", save_best_only=True),
    EarlyStopping(patience=8),
    ReduceLROnPlateau(factor=0.5, patience=3)
]

print("🚀 Iniciando entrenamiento Mask-MobileNet...")
model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

# Exportación a TFLite (Para tu App en Kotlin)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("EggplantMaskModel.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Modelo TFLite listo para integrarse en Android/Kotlin.")