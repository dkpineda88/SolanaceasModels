# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 13:52:16 2026

@author: dkpin
"""

import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. CONFIGURACIÓN ---
IMG_SIZE = (256, 256)
NUM_CLASSES = 4  # Basado en tus categorías [0, 1, 2, 3]
BATCH_SIZE = 8
EPOCHS = 25

# Rutas de tus archivos (Ajusta los nombres según tu descarga)
TRAIN_TFRECORD = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v3i.tfrecord/train/EarlyBlight.tfrecord'
VAL_TFRECORD = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v3i.tfrecord/valid/EarlyBlight.tfrecord'
TEST_TFRECORD = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v3i.tfrecord/test/EarlyBlight.tfrecord'


# --- 2. FUNCIÓN DE PARSEO (Lectura de Datos) ---
import tensorflow as tf

# --- CONFIGURACIÓN BASADA EN TU .PBTXT ---
# ID 0: Background (Fondo/Hoja sana sin manchas)
# ID 1: EarlyBlight
# ID 2: Healthy (Hoja completa sana)
# ID 3: LateBlight
NUM_CLASSES = 4 
IMG_SIZE = (256, 256)

import tensorflow as tf

def parse_proto(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/mask': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # 1. Decodificar Imagen
    image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    # Importante: Mantener el preprocesamiento de ResNet50
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    # 2. Decodificar Máscaras y Labels (Formato Sparse a Dense)
    masks = tf.sparse.to_dense(parsed_features['image/object/mask'], default_value='')
    labels = tf.sparse.to_dense(parsed_features['image/object/class/label'], default_value=0)
    
    # Inicializamos la máscara en ceros (Fondo = Canal 0)
    # Shape: (256, 256, 4)
    final_mask = tf.zeros((*IMG_SIZE, NUM_CLASSES), dtype=tf.float32)
    
    # 3. Lógica de Clasificación (Para la rama class_out)
    if tf.shape(labels)[0] > 0:
        main_label = tf.cast(labels[0], tf.int64)
    else:
        main_label = tf.cast(2, tf.int64) # ID 2 = Healthy por defecto

    # 4. Lógica de Segmentación (Para la rama mask_out)
    # Solo procesamos si hay máscaras de objetos detectados
    def process_masks():
        # Usamos un tensor local para acumular las máscaras
        temp_mask = tf.zeros((*IMG_SIZE, NUM_CLASSES), dtype=tf.float32)
        
        # Iteramos sobre cada objeto detectado en la imagen
        for i in range(tf.shape(masks)[0]):
            m = tf.io.decode_png(masks[i], channels=1)
            m = tf.image.resize(m, IMG_SIZE, method="nearest")
            
            # Convertimos el ID del pbtxt (1, 2 o 3) en su canal correspondiente
            label_idx = tf.cast(labels[i], tf.int32)
            one_hot_layer = tf.one_hot(label_idx, NUM_CLASSES) 
            
            # Solo los píxeles donde m > 0 pertenecen a la enfermedad
            mask_binary = tf.cast(m > 0, tf.float32)
            temp_mask = tf.maximum(temp_mask, mask_binary * one_hot_layer)
        return temp_mask

    # Si no hay máscaras (imagen Healthy), la máscara se queda en 0 (puro fondo)
    # Esto soluciona el problema de los 65536 píxeles
    final_mask = tf.cond(tf.shape(masks)[0] > 0, 
                         process_masks, 
                         lambda: tf.zeros((*IMG_SIZE, NUM_CLASSES), dtype=tf.float32))

    return image, {
        "mask_out": final_mask, 
        "class_out": tf.one_hot(tf.cast(main_label, tf.int32), NUM_CLASSES)
    }

# --- 3. CARGA DE DATASETS ---
def get_dataset(path):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = get_dataset(TRAIN_TFRECORD)
val_ds = get_dataset(VAL_TFRECORD)
test_ds = get_dataset(TEST_TFRECORD)

# --- 3. ARQUITECTURA DEL MODELO (U-Net + ResNet50) ---
def build_model():
    base_model = tf.keras.applications.ResNet50(
        input_shape=(*IMG_SIZE, 3), 
        include_top=False, 
        weights='imagenet'
    )
    
    # Salidas para Skip Connections
    layer_names = [
        "conv1_relu",          # 128x128
        "conv2_block3_out",    # 64x64
        "conv3_block4_out",    # 32x32
        "conv4_block6_out",    # 16x16
        "conv5_block3_out"     # 8x8
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
    
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    skips = encoder(inputs)
    x = skips[-1] # Empezamos en 8x8
    
    # Rama de Clasificación
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    class_out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='class_out')(gap)
    
    # Decoder (Subida)
    # Lista de filtros para el decoder
    decoder_filters = [256, 128, 64, 32]
    
    for i in range(len(decoder_filters)):
        # Subimos la resolución x2
        x = tf.keras.layers.Conv2DTranspose(decoder_filters[i], (2, 2), strides=(2, 2), padding='same')(x)
        # Conexión de salto (Skip connection)
        skip_idx = len(skips) - 2 - i
        if skip_idx >= 0:
            x = tf.keras.layers.Concatenate()([x, skips[skip_idx]])
        
        x = tf.keras.layers.Conv2D(decoder_filters[i], (3, 3), activation='relu', padding='same')(x)

    # CAPA FINAL DE RECONSTRUCCIÓN: Asegura llegar a 256x256
    # Si 'conv1_relu' es 128x128, necesitamos un último upsampling
    x = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    
    mask_out = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='mask_out')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=[mask_out, class_out])

model = build_model()

# --- 4. COMPILACIÓN CON PESOS ---
# Le damos peso 100 a la clasificación para que deje de estar en 34% rápidamente
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    # Solo calculamos sobre los canales de enfermedad (0 y 2)
    y_true_f = K.flatten(y_true[..., 0] + y_true[..., 2])
    y_pred_f = K.flatten(y_pred[..., 0] + y_pred[..., 2])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss={
        'mask_out': 'categorical_crossentropy', 
        'class_out': 'categorical_crossentropy'
    },
    # Subimos el peso de la máscara de 1.0 a 100.0
# =============================================================================
     loss_weights={
         'mask_out': 0.1, 
         'class_out': 5.0
     },
# =============================================================================
    metrics={"mask_out": "accuracy", "class_out": "accuracy"}
)

early_stop = EarlyStopping(
    monitor='val_loss',      # Vigila la pérdida de validación
    patience=5,             # Si después de 2 épocas no mejora, se detiene
    restore_best_weights=True, # Al terminar, se queda con los mejores pesos (vital)
    verbose=1
)

# Recomendación extra: Guardar el mejor modelo automáticamente
checkpoint = ModelCheckpoint(
    'mejor_modelo_unet.keras', 
    monitor='val_loss', 
    save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_mask_out_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-7,
    verbose=1
)


# --- 5. ENTRENAMIENTO ---
model.fit(train_ds, validation_data=val_ds, epochs=25,callbacks=[early_stop, checkpoint,reduce_lr] )

results = model.evaluate(test_ds)
print(f"Total Loss: {results[0]:.4f}")
print(f"Segmentación Accuracy: {results[3]*100:.2f}%")
print(f"Clasificación Accuracy: {results[4]*100:.2f}%")

import matplotlib.pyplot as plt
import numpy as np

def visualize_results(dataset, model, num_samples=3):
    for images, targets in dataset.take(1):
        preds = model.predict(images)
        pred_masks = preds[0]  
        pred_classes = preds[1] 

        plt.figure(figsize=(15, 5 * num_samples))
        
        for i in range(num_samples):
            # 1. IMAGEN ORIGINAL: Normalización para visualización
            img = images[i].numpy()
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

            # 2. MÁSCARA REAL (MODO BINARIO): Forzamos visibilidad
            # Sumamos canales 1, 2 y 3. Si un píxel tiene 0.001, se volverá visible.
            gt_data = targets['mask_out'][i].numpy()[..., 1:]
            gt_combined = np.any(gt_data > 0, axis=-1).astype(np.float32)
            
            # Debug por consola para estar 100% seguros
            print(f"Muestra {i}: Píxeles con enfermedad detectados: {np.sum(gt_combined)}")

            # 3. MÁSCARA PREDICHA (MODO CALOR): 
            # Mostramos la probabilidad máxima de CUALQUIER canal de enfermedad
            p_mask_disease = np.max(pred_masks[i][..., 1:], axis=-1)
            
            # 4. CLASIFICACIÓN
            gt_class = np.argmax(targets['class_out'][i])
            p_class = np.argmax(pred_classes[i])
            conf = np.max(pred_classes[i])

            # --- PLOTEO ---
            # Imagen Original
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(img)
            plt.title(f"Original - Clase Real: {gt_class}")
            plt.axis('off')

            # Máscara Real: Ahora en Blanco y Negro (Blanco = Enfermedad)
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(gt_combined, cmap='gray') 
            plt.title(f"Real (Píxeles: {int(np.sum(gt_combined))})")
            plt.axis('off')

            # Predicción: Mapa de calor 'hot' (Blanco/Amarillo = Alta confianza)
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(p_mask_disease, cmap='hot') 
            plt.title(f"Pred: {p_class} ({conf:.2%})")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

# Ejecutar la visualización
visualize_results(val_ds, model)




for images, targets in train_ds.take(1):
    mask_data = targets['mask_out'].numpy()
    print("¿Hay píxeles de enfermedad en el Ground Truth?:", np.any(mask_data[..., 1:] > 0))
    print("Valor máximo en la máscara real:", np.max(mask_data))
