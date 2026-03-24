# -*- coding: utf-8 -*-
"""
Optimized for Android: SIS-YOLOv8 + Triple Stage (ASPP) + COCO JSON
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, ops
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. PARÁMETROS ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 25
NUM_CLASSES = 9  # Definido para tus 9 clases del JSON

# --- 2. GENERADOR DE DATOS (Mapeo de Clases Corregido) ---
class SolanaceaeMultiOutputDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256), num_classes=9):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Mapeo robusto: Algunos JSON COCO tienen IDs saltados (1, 2, 10...)
        # Esto mapea ID_REAL_JSON -> INDICE_MODELO (0 a 8)
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_map = {cat_id: i for i, cat_id in enumerate(cat_ids[:num_classes])}
        print(f"Mapeo de Clases (COCO ID -> Modelo Index): {self.cat_map}")

    def __len__(self):
        return len(self.ids) // self.batch_size

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                path = os.path.join(self.img_dir, img_info['file_name'])
                image = cv2.imread(path)
                if image is None: continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.img_size)

                mask = np.zeros((*self.img_size, self.num_classes), dtype=np.float32)
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                
                main_cat_idx = 0 
                if anns:
                    coco_cat_id = anns[0]['category_id']
                    main_cat_idx = self.cat_map.get(coco_cat_id, 0)

                for ann in anns:
                    coco_cat_id = ann['category_id']
                    cat_idx = self.cat_map.get(coco_cat_id)
                    
                    if cat_idx is not None and cat_idx < self.num_classes:
                        # --- VALIDACIÓN CRÍTICA PARA EVITAR EL INDEXERROR ---
                        # Verificamos que 'segmentation' exista, no sea nulo y tenga datos
                        seg = ann.get('segmentation', None)
                        if seg is None or len(seg) == 0 or (isinstance(seg, list) and len(seg[0]) == 0):
                            continue # Ignora anotaciones sin polígono real

                        try:
                            m = self.coco.annToMask(ann)
                            m = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            mask[:, :, cat_idx] = np.maximum(mask[:, :, cat_idx], m)
                        except Exception:
                            continue # Si falla la conversión de esta máscara, sigue con la siguiente

                X.append(preprocess_input(image.astype(np.float32)))
                y_mask.append(mask)
                y_class.append(tf.keras.utils.to_categorical(main_cat_idx, num_classes=self.num_classes))

            except Exception as e:
                print(f"Error procesando imagen ID {img_id}: {e}")
                continue

        # Si por errores el batch queda vacío, devolvemos el siguiente (recursión simple)
        if len(X) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        return np.array(X), {"mask_out": np.array(y_mask), "class_out": np.array(y_class)}
# --- 3. ARQUITECTURA (Triple Etapa: ResNet50 + ASPP) ---


def aspp_block(x, filters):
    shape = ops.shape(x)
    b0 = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation('swish')(b0)
    
    branches = [b0]
    for rate in [6, 12, 18]:
        b = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=rate, use_bias=False)(x)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('swish')(b)
        branches.append(b)
        
    pool = layers.GlobalAveragePooling2D()(x)
    pool = layers.Reshape((1, 1, ops.shape(x)[-1]))(pool)
    pool = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(pool)
    pool = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation='bilinear')(pool)
    branches.append(pool)
    
    out = layers.Concatenate()(branches)
    out = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(out)
    return layers.Activation('swish')(out)

def build_triple_stage_model(num_classes=9):
    base_model = tf.keras.applications.ResNet50(input_shape=(256,256,3), include_top=False, weights='imagenet')
    
    s1 = base_model.input                                     
    s2 = base_model.get_layer("conv1_relu").output            
    s3 = base_model.get_layer("conv2_block3_out").output      
    s4 = base_model.get_layer("conv3_block4_out").output      
    bridge = base_model.get_layer("conv4_block6_out").output  

    x = aspp_block(bridge, 256)

    def decoder_block(inputs, skip, filters):
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="swish")(x)
        x = layers.BatchNormalization()(x)
        return x

    d1 = decoder_block(x, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(d4)

    gap = layers.GlobalAveragePooling2D()(base_model.output)
    fc = layers.Dense(512, activation='swish')(gap)
    fc = layers.Dropout(0.4)(fc)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(fc)

    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])

# --- 4. COMPILACIÓN Y ENTRENAMIENTO ---
model = build_triple_stage_model(num_classes=NUM_CLASSES)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"mask_out": dice_loss, "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 20.0, "class_out": 1.0},
    metrics={"mask_out": ["accuracy"], "class_out": ["accuracy"]}
)

# Cargar Datasets (Ajusta tus rutas)
base_path = 'D:/DATASETS/Imagenes/Solanaceas/Tomato Leaf Disease.v6i.coco/'
train_ds = SolanaceaeMultiOutputDataset(base_path + "train", base_path + "train/_annotations.coco.json", num_classes=NUM_CLASSES)
val_ds = SolanaceaeMultiOutputDataset(base_path + "valid", base_path + "valid/_annotations.coco.json", num_classes=NUM_CLASSES)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint('mejor_modelo_triple.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_mask_out_loss', factor=0.5, patience=3)
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- 5. EXPORTACIÓN A TFLITE (Optimizado para Android) ---
model.save("msolUNETResNetCoco3.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('msolUNETResNetCoco3.tflite', 'wb') as f:
    f.write(tflite_model)
print("Conversión TFLite Exitosa.")