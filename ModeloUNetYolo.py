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

class YOLOOBBDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, label_dir, batch_size=4, img_size=(256, 256), shuffle=True):
        """
        Args:
            img_dir: Carpeta de imágenes.
            label_dir: Carpeta de archivos .txt de YOLO OBB.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Listar nombres de archivos (asumiendo que imagen y txt se llaman igual)
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if self.shuffle:
            np.random.shuffle(self.img_files)
    
    def __len__(self):
        return int(np.floor(len(self.img_files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_filenames = self.img_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []
        
        target_h, target_w = self.img_size 
        
        for filename in batch_filenames:
            # 1. Cargar Imagen
            img_path = os.path.join(self.img_dir, filename)
            img_orig = cv2.imread(img_path)
            if img_orig is None: continue 
            
            img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h)) 
            img = img / 255.0
            
            # 2. Cargar Etiqueta OBB (.txt)
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 9: continue # Debe tener class_id + 8 coordenadas
                        
                        # Las coordenadas YOLO OBB están normalizadas (0 a 1)
                        # Formato: class x1 y1 x2 y2 x3 y3 x4 y4
                        points = np.array(parts[1:], dtype=np.float32).reshape((4, 2))
                        
                        # ESCALADO DIRECTO: Multiplicar por el tamaño del lienzo
                        points[:, 0] *= target_w  # Escala X
                        points[:, 1] *= target_h  # Escala Y
                        
                        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            
            images.append(img)
            masks.append(mask)
        
        return np.array(images), np.expand_dims(np.array(masks), axis=-1)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_files)


# Definir parámetros (Ajusta las carpetas de 'labels')
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/'

train_dataset = YOLOOBBDataset(
    img_dir=base_path + 'train/images/', 
    label_dir=base_path + 'train/labels/', 
    batch_size=4
)

valid_dataset = YOLOOBBDataset(
    img_dir=base_path + 'valid/images/', 
    label_dir=base_path + 'valid/labels/', 
    batch_size=4
)

test_dataset = YOLOOBBDataset(
    img_dir=base_path + 'test/images/', 
    label_dir=base_path + 'test/labels/', 
    batch_size=4
)

##Graficar imagen real+ mascara + segmentacion
for batch_images, batch_masks in train_dataset:
    for i in range(len(batch_images)):
        img = batch_images[i]
        # Aseguramos que la máscara sea binaria pura (0 o 1)
        mask = np.squeeze(batch_masks[i]).astype(np.float32)
        
        # DEBUG: Verifica si la máscara tiene algún píxel blanco
        if np.max(mask) == 0:
            print(f"Advertencia: La máscara {i} está vacía (todo negro).")
        
        # 1. Crear la parte segmentada (Imagen * Máscara)
        # Expandimos la máscara a 3 canales para que coincida con RGB (256, 256, 3)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        segmented_img = img * mask_3d

        # 2. Visualización
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Imagen Original
        axes[0].imshow(img)
        axes[0].set_title("Imagen Original")
        axes[0].axis('off')

        # Máscara (Forzamos visualización escalando 0-1)
        axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Máscara (Max valor: {np.max(mask)})")
        axes[1].axis('off')

        # Resultado de la Segmentación
        axes[2].imshow(segmented_img)
        axes[2].set_title("Parte Segmentada")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    break




# Crear el modelo de Keras (ejemplo básico de U-Net)
inputs = tf.keras.layers.Input((256, 256, 3))
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D()(x)

x = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

model = tf.keras.models.Model(inputs, x)



def build_unet(input_shape=(256, 256, 3)):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder (Contracción)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    
    # Decoder (Expansión) + SKIP CONNECTION
    u3 = tf.keras.layers.UpSampling2D((2, 2))(c2)
    # Concatenamos la salida de c1 con la subida de u3
    concat3 = tf.keras.layers.Concatenate()([u3, c1]) 
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat3)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)
    
    return tf.keras.models.Model(inputs, outputs)

model = build_unet()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryFocalCrossentropy(), # Mejor para detectar objetos pequeños
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=3)]
)

# Compilar el modelo
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
# Entrenar el modelo con el dataset de validación incluido
history = model.fit(
    train_dataset,              # Generador de entrenamiento
    epochs=10,                  # Número de épocas
    validation_data=valid_dataset, # Generador de validación
    #workers=4,                  # (Opcional) Mejora velocidad usando múltiples hilos
    #use_multiprocessing=False   # Cambiar a True si estás en Linux para mayor velocidad
)

results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")


# 0. Definir un diccionario con tus clases (ajusta según tu dataset de Roboflow)
# El ID 0 suele ser la primera clase que aparece en tu archivo data.yaml
class_names = {0: "EarlyBlight", 1: "Healthy", 2: "LateBlight"} 

# Iterar sobre los batches del dataset
for batch_idx in range(len(test_dataset)):
    batch_images, batch_masks = test_dataset[batch_idx]
    preds = model.predict(batch_images, verbose=0)
    
    # Obtener los nombres de archivos para este batch (necesario para buscar el .txt)
    start_idx = batch_idx * test_dataset.batch_size
    batch_files = test_dataset.img_files[start_idx : start_idx + test_dataset.batch_size]

    for i in range(len(batch_images)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # --- Obtener la etiqueta (enfermedad) desde el archivo .txt ---
        filename = batch_files[i]
        label_path = os.path.join(test_dataset.label_dir, os.path.splitext(filename)[0] + '.txt')
        enfermedad = "Desconocida"
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                first_line = f.readline()
                if first_line:
                    class_id = int(first_line.split()[0])
                    enfermedad = class_names.get(class_id, f"Clase {class_id}")

        # 1. IMAGEN ORIGINAL
        axes[0].imshow(batch_images[i])
        axes[0].set_title(f"Imagen: {filename}")
        axes[0].axis('off')

        # 2. MÁSCARA REAL
        axes[1].imshow(np.squeeze(batch_masks[i]), cmap='gray')
        axes[1].set_title(f"Diagnóstico Real:\n{enfermedad}")
        axes[1].axis('off')

        # 3. PREDICCIÓN DEL MODELO
        pred_binaria = (np.squeeze(preds[i]) > 0.5).astype(np.uint8)
        
        # Calculamos si el modelo detectó algo (píxeles > 0)
        tiene_enfermedad = np.max(pred_binaria) > 0
        resultado_pred = enfermedad if tiene_enfermedad else "No detectado"

        axes[2].imshow(pred_binaria, cmap='gray')
        axes[2].set_title(f"Predicción U-Net:\n{resultado_pred}")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    
   # break # Solo un batch para probar

model_version = 1
# Guardar el modelo en formato Keras (.keras)
model.save("msolUNET.keras")

import os
print(os.getcwd())  # Esto imprimirá el directorio actual de trabajo


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

open("msolUNET.tflite","wb").write(tfmodel)
