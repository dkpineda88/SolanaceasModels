# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:50:27 2026


@author: dkpin
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A

class YOLOOBBDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, label_dir, batch_size=4, img_size=(256, 256), shuffle=True, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Número de clases fijo para tu dataset
        self.num_classes = 3

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_filenames = self.img_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks, labels = [], [], []
        target_h, target_w = self.img_size 

        for filename in batch_filenames:
            img_path = os.path.join(self.img_dir, filename)
            img = cv2.imread(img_path)
            if img is None: continue 
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h))
            
            # 1. Máscara de 3 canales (H x W x 3)
            mask = np.zeros((target_h, target_w, self.num_classes), dtype=np.float32)
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')
            
            # Clase dominante (la que tenga más área o la primera encontrada)
            main_class_id = 0 

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.split()
                        if len(parts) < 9: continue
                        
                        try:
                            current_class_id = int(parts[0])
                            # Aseguramos que el ID esté dentro del rango 0, 1, 2
                            if current_class_id >= self.num_classes: continue
                            
                            main_class_id = current_class_id
                            
                            # 2. Extraer y escalar puntos (OBB)
                            points = np.array(parts[1:9], dtype=np.float32).reshape((4, 2))
                            points[:, 0] *= target_w
                            points[:, 1] *= target_h
                            
                            # 3. Dibujar en el canal específico
                            temp_mask = np.zeros((target_h, target_w), dtype=np.float32)
                            cv2.fillPoly(temp_mask, [points.astype(np.int32)], 1.0)
                            
                            # Combinar con lo que ya haya en ese canal
                            mask[:, :, current_class_id] = np.maximum(mask[:, :, current_class_id], temp_mask)
                        except:
                            continue

            # 4. Normalización
            images.append(img / 255.0)
            masks.append(mask)
            # Categorización de la clase principal para la salida de clasificación
            labels.append(tf.keras.utils.to_categorical(main_class_id, num_classes=self.num_classes))

        return np.array(images, dtype=np.float32), {
            "mask_out": np.array(masks, dtype=np.float32), 
            "class_out": np.array(labels, dtype=np.float32)
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_files)


# Definir parámetros (Ajusta las carpetas de 'labels')
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/'
train_dataset = YOLOOBBDataset(base_path + 'train/images/', base_path + 'train/labels/')
valid_dataset = YOLOOBBDataset(base_path + 'valid/images/', base_path + 'valid/labels/')
test_dataset = YOLOOBBDataset(base_path + 'test/images/', base_path + 'test/labels/')

##Graficar imagen real+ mascara + segmentacion
# Iterar sobre los batches del dataset
for batch_images, batch_targets in train_dataset:
    batch_masks = batch_targets["mask_out"]
    batch_labels = batch_targets["class_out"]
    
    for i in range(len(batch_images)):
        img = batch_images[i]
        # La máscara ya es (256, 256, 3)
        mask = batch_masks[i]
        
        # Para visualizar, creamos una máscara 2D que tenga '1' donde haya cualquier enfermedad
        mask_2d = np.max(mask, axis=-1) 
        
        # Identificar la clase
        class_id = np.argmax(batch_labels[i])
        class_name = {0: "EarlyBlight", 1: "Healthy", 2: "LateBlight"}[class_id]
        
        # 1. Crear la parte segmentada correctamente
        # Multiplicamos la imagen (256,256,3) por la máscara expandida (256,256,1)
        segmented_img = img * mask_2d[:, :, np.newaxis]
        
        # 2. Visualización
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img)
        axes[0].set_title(f"Imagen Original ({class_name})")
        axes[0].axis('off')
        
        # Mostramos la máscara multicanal como imagen RGB para ver los colores de las enfermedades
        axes[1].imshow(mask) 
        axes[1].set_title(f"Máscara Multicanal (Max: {np.max(mask):.2f})")
        axes[1].axis('off')
        
        axes[2].imshow(segmented_img)
        axes[2].set_title("Parte Segmentada (Filtro)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    break


import tensorflow as tf

def build_unet_multi_output(input_shape=(256, 256, 3), num_classes=3):
    inputs = tf.keras.layers.Input(input_shape)

    # --- ENCODER ---
    def conv_block(x, filters, dropout_rate=0.0):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Agregamos dropout al bloque si se especifica
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    # Encoder progresivo
    f1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(f1)

    f2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(f2)

    # Empezamos a usar Dropout en las capas más densas
    f3 = conv_block(p2, 128, dropout_rate=0.1)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(f3)

    # --- BOTTLENECK (Máximo Dropout aquí) ---
    b = conv_block(p3, 256, dropout_rate=0.3)

    # --- RAMA 1: SEGMENTACIÓN (DECODER) ---
    u3 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u3 = tf.keras.layers.concatenate([u3, f3])
    c3 = conv_block(u3, 128, dropout_rate=0.1)

    u4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = tf.keras.layers.concatenate([u4, f2])
    c4 = conv_block(u4, 64)

    u5 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = tf.keras.layers.concatenate([u5, f1])
    c5 = conv_block(u5, 32)

    mask_output = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name="mask_out")(c5)

    # --- RAMA 2: CLASIFICACIÓN ---
    gap = tf.keras.layers.GlobalAveragePooling2D()(b)
    d1 = tf.keras.layers.Dense(256, activation='relu')(gap)
    d1 = tf.keras.layers.Dropout(0.5)(d1) # Dropout fuerte para clasificación
    class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name="class_out")(d1)

    return tf.keras.models.Model(inputs=inputs, outputs=[mask_output, class_output])

model_tomato = build_unet_multi_output()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuración de EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',      # Vigila la pérdida de validación
    patience=2,             # Si después de 2 épocas no mejora, se detiene
    restore_best_weights=True, # Al terminar, se queda con los mejores pesos (vital)
    verbose=1
)

# Recomendación extra: Guardar el mejor modelo automáticamente
checkpoint = ModelCheckpoint(
    'mejor_modelo_unet.keras', 
    monitor='val_loss', 
    save_best_only=True
)


model_tomato.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss={
        "mask_out": "binary_crossentropy",
        "class_out": "categorical_crossentropy"
    },
# =============================================================================
# =============================================================================
#     loss_weights={
#          "mask_out": 0.5, 
#          "class_out": 15.0 # Le damos un poco más de importancia a la máscara
#      },
# =============================================================================
# =============================================================================
# =============================================================================
    metrics={"mask_out": "accuracy", "class_out": "accuracy"}
)

# Compilar el modelo
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
# Entrenar el modelo con el dataset de validación incluido
history_tomato = model_tomato.fit(
    train_dataset,              # Generador de entrenamiento
    epochs=20,                  # Número de épocas
    validation_data=valid_dataset, # Generador de validación
    callbacks=[early_stop, checkpoint]
   
)

results = model_tomato.evaluate(test_dataset)
print(f"Total Loss: {results[0]:.4f}")
print(f"Segmentación Accuracy: {results[3]*100:.2f}%")
print(f"Clasificación Accuracy: {results[4]*100:.2f}%")


# 0. Definir un diccionario con tus clases (ajusta según tu dataset de Roboflow)
# El ID 0 suele ser la primera clase que aparece en tu archivo data.yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Diccionario de configuración para los 3 canales de tu salida mask_out
# Canal 0: EarlyBlight, Canal 1: Healthy, Canal 2: LateBlight
class_info = {
    0: {"name": "Early Blight", "color": (255, 0, 0)},  # Rojo
    1: {"name": "Healthy", "color": (0, 255, 0)},      # Verde
    2: {"name": "Late Blight", "color": (0, 0, 255)}    # Azul
}

def create_roboflow_viz(image, mask_multichannel):
    # Convertir imagen a formato OpenCV (0-255 uint8)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    output = image.copy()
    overlay = image.copy()
    
    # Iteramos sobre cada canal de la máscara (0, 1, 2)
    for class_id in range(mask_multichannel.shape[-1]):
        # Si es la clase 'Healthy' (1), quizás quieras saltarla para no tapar toda la hoja
        if class_id == 1: continue 
            
        # Máscara binaria para el canal actual
        m = (mask_multichannel[:, :, class_id] > 0.1).astype(np.uint8)
        # Pon esto antes de pintar
        for c in range(3):
          print(f"Canal {c} - Max valor: {np.max(mask_multichannel[:,:,c]):.4f}")
        
        if np.any(m):
            color = class_info[class_id]["color"]
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 1. Dibujar relleno en el overlay
            cv2.fillPoly(overlay, contours, color)
            
            # 2. Dibujar bordes sólidos en la imagen de salida
            cv2.drawContours(output, contours, -1, color, 2)

    # Mezclar la imagen original con el overlay transparente (alpha=0.4)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

# --- EJECUCIÓN DEL BATCH ---
for batch_idx in range(len(test_dataset)):
    batch_images, batch_targets = test_dataset[batch_idx]
    
    # Realizar predicción
    # preds[0] es 'mask_out' (B, 256, 256, 3), preds[1] es 'class_out' (B, 3)
    preds = model_tomato.predict(batch_images, verbose=0)
    
    for i in range(len(batch_images)):
        img_orig = batch_images[i]
        mask_pred = preds[0][i]
        class_pred = preds[1][i]
        
        # Generar visualización estilo Roboflow
        viz_result = create_roboflow_viz(img_orig, mask_pred)
        
        # Info de clasificación para el título
        idx_pred = np.argmax(class_pred)
        conf = class_pred[idx_pred] * 100
        nombre = class_info[idx_pred]["name"]
        
        # Mostrar resultados
        plt.figure(figsize=(10, 10))
        plt.imshow(viz_result)
        plt.title(f"Predicción: {nombre} ({conf:.1f}%)\nColores: Rojo=Early, Azul=Late", fontsize=14)
        plt.axis('off')
        plt.show()
    
    break # Solo mostramos el primer batch



import tensorflow as tf

# Cargar el mejor modelo guardado
model = tf.keras.models.load_model("msolUNET3.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 1. Intentar optimizar para que no necesite FLEX si es posible
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 2. Habilitar compatibilidad con Flex (necesaria por tu Conv2D)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS 
]

# 3. Importante: Forzar que no use versiones de API muy nuevas
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()

with open("msolUNET3.tflite", "wb") as f:
    f.write(tflite_model)



model_version = 1
# Guardar el modelo en formato Keras (.keras)
model.save("msolUNET4.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ESTA ES LA LÍNEA CLAVE: Forzar compatibilidad con versiones anteriores
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Operaciones estándar de TFLite
    tf.lite.OpsSet.SELECT_TF_OPS    # Permite operaciones de TF original si es necesario
]

# Opcional: Desactiva optimizaciones experimentales que suben la versión de los opcodes
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

with open("msolUNET4.tflite", "wb") as f:
    f.write(tflite_model)
    
open("msolUNET4.tflite","wb").write(tflite_model)
