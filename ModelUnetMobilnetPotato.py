# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:15:38 2026

@author: dkpin
"""

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
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURACIÓN GLOBAL (Filosofía Pepper)
# ============================================================================
IMG_SIZE = (224, 224) # 224 es el estándar nativo de MobileNetV2 (más rápido)
BATCH_SIZE = 16
CLASS_NAMES = ["EarlyBlight", "Healthy", "LateBlight"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# DATASET ADAPTADO (Filosofía Pepper)
# ============================================================================
class PotatoDiseaseDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=16, img_size=(224, 224), is_train=True, **kwargs):
        # SOLUCIÓN AL WARNING: Llamar al constructor padre
        super().__init__(**kwargs)
        
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_train = is_train
        
        # Mapeo: Asegúrate de que estos IDs existan en tu archivo .json
        self.cat_map = {1: 0, 2: 1, 3: 2} 
        self.num_classes = 3
        self.healthy_idx = 1 
        
    # --- ESTA ES LA FUNCIÓN QUE FALTA ---
    def __len__(self):
        # Calcula el número de batches por época
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                image = cv2.imread(os.path.join(self.img_dir, img_info['file_name']))
                if image is None: continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.img_size)

                mask = np.zeros((self.img_size[1], self.img_size[0], self.num_classes), dtype=np.float32)
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)

                current_class_idx = self.healthy_idx 

                if anns:
                    # VALIDACIÓN: Evitar el "index out of range" si la lista existe pero está vacía
                    raw_cat = anns[0].get('category_id', 2)
                    current_class_idx = self.cat_map.get(raw_cat, self.healthy_idx)

                    for ann in anns:
                        m_id = self.cat_map.get(ann['category_id'], -1)
                        # VALIDACIÓN EXTRA: Verificar que 'segmentation' no esté vacío
                        if 0 <= m_id < self.num_classes and ann.get('segmentation'):
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, m_id] = np.maximum(mask[:, :, m_id], m.astype(np.float32))

                if np.max(mask) == 0:
                    mask[:, :, self.healthy_idx] = 1.0

                X.append(preprocess_input(image.astype(np.float32)))
                y_mask.append(mask)
                y_class.append(tf.keras.utils.to_categorical(current_class_idx, num_classes=self.num_classes))
            
            except Exception:
                # Silenciamos los errores repetitivos para no saturar la consola
                continue

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}
# ============================================================================
# 1. ARQUITECTURA ULTRA-LITE (MobileNetV2 + Separable Decoder)
# ============================================================================
def build_fast_unet_potato(input_shape=(224, 224, 3), num_classes=3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    
    base_model.trainable = True # Permitimos fine-tuning suave

    # Skip connections estratégicas
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
        # Filosofía Pepper: SeparableConv para bajar parámetros y latencia
        x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    # Decoder progresivo
    x = lite_decoder_block(bridge, skips[3], 256) # 14x14
    x = lite_decoder_block(x, skips[2], 128)      # 28x28
    x = lite_decoder_block(x, skips[1], 64)       # 56x56
    x = lite_decoder_block(x, skips[0], 32)       # 112x112
    
    # Salida Final Segmentación
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x) # 224x224
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(x)

    # Clasificación Paralela (Global Average Pooling)
    gap = layers.GlobalAveragePooling2D()(bridge)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(gap)

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# ============================================================================
# 2. COMBO LOSS (CrossEntropy + Dice) - CRÍTICO PARA mIoU
# ============================================================================
def combo_loss(y_true, y_pred):
    # Categorical Crossentropy estándar
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Dice Loss para mejorar mIoU en clases pequeñas (enfermedades)
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f, 0) + tf.reduce_sum(y_pred_f, 0) + 1e-6)
    return ce + (1 - tf.reduce_mean(dice))

# ============================================================================
# 3. COMPILACIÓN Y ENTRENAMIENTO
# ============================================================================
model = build_fast_unet_potato()

# Learning rate ligeramente más alto para romper el estancamiento
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    """Calcula el IoU convirtiendo automáticamente One-Hot a Índices"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=1.33):
    """
    Especial para datasets desbalanceados (manchas pequeñas).
    Penaliza falsos negativos más que falsos positivos.
    """
    def loss(y_true, y_pred):
        # Ajustado a NUM_CLASSES_SEG
        y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
        
        tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
        
        tvi = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
        return tf.reduce_mean(tf.pow(1.0 - tvi, gamma))
    return loss


def combined_seg_loss(alpha=0.7, beta=0.3, gamma=1.33,
                      w_tversky=0.6, w_ce=0.4):
    ftl = focal_tversky_loss(alpha, beta, gamma)
    def loss(y_true, y_pred):
        return (w_tversky * ftl(y_true, y_pred)
                + w_ce * tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_true, y_pred)))
    return loss



model.compile(
    optimizer=optimizer,
    loss={"mask_out": combo_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 5.0, "class_out": 0.5}, # Priorizamos la máscara
    metrics={
        "mask_out":  [UpdatedMeanIoU(num_classes=NUM_CLASSES, name="miou"),
                      "accuracy"],
        "class_out": "accuracy"
    }
)

# ============================================================================
# 4. DATASET (Ajustado a 224x224)
# ============================================================================
# Reutiliza tu clase PotatoDiseaseDataset pero cambia img_size=(224, 224)
base_path = 'D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v5i.coco/'
train_ds = PotatoDiseaseDataset(base_path + "train", base_path + "train/_annotations.coco.json", img_size=(224, 224), batch_size=BATCH_SIZE)
val_ds = PotatoDiseaseDataset(base_path + "valid", base_path + "valid/_annotations.coco.json", img_size=(224, 224), batch_size=BATCH_SIZE)
test_ds = PotatoDiseaseDataset(base_path + "test", base_path + "test/_annotations.coco.json", img_size=(224, 224), batch_size=BATCH_SIZE)


# ============================================================================
# CALLBACKS & RUN
# ============================================================================
callbacks = [
    EarlyStopping(monitor='val_mask_out_miou', patience=12, mode='max', restore_best_weights=True),
    ModelCheckpoint('best_potato_pepper_style.keras', monitor='val_mask_out_miou', save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_mask_out_miou', factor=0.2, patience=5, mode='max', min_lr=1e-7)
]


history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=50, 
    callbacks=callbacks
)

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
