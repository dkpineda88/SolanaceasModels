# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 11:30:24 2026

@author: dkpin
"""

import tensorflow as tf
import numpy as np
import json
import os
import cv2
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import datetime


class CocoDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=4, img_size=(256, 256), shuffle=True):
        """
        Args:
            img_dir (string): Directorio de imágenes.
            ann_file (string): Ruta al archivo de anotaciones JSON.
            batch_size (int, optional): Tamaño del batch.
            img_size (tuple, optional): Tamaño de las imágenes de entrada (alto, ancho).
            shuffle (bool, optional): Si se debe mezclar los datos.
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Cargar el archivo COCO
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        self.img_ids = [image['id'] for image in self.coco['images']]
        self.img_info = {image['id']: image for image in self.coco['images']}
        self.annotations = {image_id: [] for image_id in self.img_ids}
        
        # Organizar las anotaciones por imagen
        for ann in self.coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        
        if self.shuffle:
            np.random.shuffle(self.img_ids)
    
    def __len__(self):
        """ Número de batches por época """
        return int(np.floor(len(self.img_ids) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_ids = self.img_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []
        
        # 1. Extraer dimensiones del img_size definido en __init__
        target_h, target_w = self.img_size 
        
        for img_id in batch_ids:
            img_info = self.img_info[img_id]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            img_orig = cv2.imread(img_path)
            if img_orig is None: continue 
            
            h_orig, w_orig = img_orig.shape[:2]
            
            # 2. CORRECCIÓN: cv2.resize usa (ANCHO, ALTO)
            img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h)) 
            img = img / 255.0
            
            # 3. CORRECCIÓN: Crear máscara con dimensiones correctas
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            anns = self.annotations[img_id]
            
            for ann in anns:
                if 'segmentation' in ann:
                    seg = ann['segmentation']
                    if isinstance(seg, list):
                        for poly_list in seg:
                            # 4. CORRECCIÓN: Reshape flexible y conversión a float para escalar
                            poly = np.array(poly_list).reshape((-1, 2)).astype(np.float32)
                            
                            # 5. CORRECCIÓN: Escalar usando dimensiones originales vs target
                            poly[:, 0] *= (target_w / w_orig) # Eje X
                            poly[:, 1] *= (target_h / h_orig) # Eje Y
                            
                            # 6. CORRECCIÓN: Asegurar que sean enteros para OpenCV
                            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                    
                    elif isinstance(seg, dict):
                        from pycocotools import mask as maskUtils
                        rle_mask = maskUtils.decode(seg)
                        # Resize de máscara RLE (usar INTER_NEAREST para no perder etiquetas)
                        rle_mask = cv2.resize(rle_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        mask = np.maximum(mask, rle_mask)
    
            images.append(img)
            masks.append(mask)
        
        return np.array(images), np.expand_dims(np.array(masks), axis=-1)



    
    def on_epoch_end(self):
        """ Opcional: mezclar los datos al final de cada época """
        if self.shuffle:
            np.random.shuffle(self.img_ids)

# Definir parámetros
img_dir = 'D:/DATASETS/Imagenes/Solanaceas/TomatoCoco/train/'
ann_file = 'D:/DATASETS/Imagenes/Solanaceas/TomatoCoco/train/_annotations.coco.json'

X_valid = 'D:/DATASETS/Imagenes/Solanaceas/TomatoCoco/valid/_annotations.coco.json'
img_dir_valid = 'D:/DATASETS/Imagenes/Solanaceas/TomatoCoco/valid/'

X_test = 'D:/DATASETS/Imagenes/Solanaceas/TomatoCoco/test/_annotations.coco.json'
img_dir_test = 'D:/DATASETS/Imagenes/TomatoCoco/Tomato/test/'





# Crear el dataset
train_dataset = CocoDataset(img_dir=img_dir, ann_file=ann_file, batch_size=4, img_size=(256, 256), shuffle=True)
valid_dataset = CocoDataset(img_dir=img_dir_valid, ann_file=X_valid, batch_size=4, img_size=(256, 256), shuffle=True)
test_dataset = CocoDataset(img_dir=img_dir_test, ann_file=X_test, batch_size=4, img_size=(256, 256), shuffle=True)


# Verifica si la primera imagen tiene anotaciones asociadas
test_id = train_dataset.img_ids[0]
print(f"ID de imagen: {test_id}")
print(f"Anotaciones encontradas para este ID: {len(train_dataset.annotations[test_id])}")

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
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D()(x)
x = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

model = tf.keras.models.Model(inputs, x)

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

print(f"Imágenes cargadas: {len(test_dataset.img_ids)}")
# Probar a obtener un solo elemento manualmente
img_batch, mask_batch = test_dataset[0]
print(f"Forma del batch: {img_batch.shape}") 


def visualize_prediction(dataset, model, n_images=3):
    # ERROR CORREGIDO: Extraer el primer batch del generador
    # dataset[0] devuelve la tupla (batch_images, batch_masks)
    images, masks = dataset[0] 
    
    # Realizar la predicción sobre ese batch
    preds = model.predict(images, verbose=0)
    preds_thresholded = (preds > 0.5).astype(np.uint8)

    plt.figure(figsize=(15, 5 * n_images))
    
    for i in range(min(n_images, len(preds))):
        # Mostrar Imagen Original
        plt.subplot(n_images, 3, i*3 + 1)
        pred = np.squeeze(preds[i])
        plt.imshow(pred)
        plt.title(f"Imagen {i+1}")
        plt.axis('off')

        # Mostrar Máscara Real
        plt.subplot(n_images, 3, i*3 + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray') # squeeze elimina el canal extra (1)
        plt.title("Máscara Real")
        plt.axis('off')

        # Mostrar Predicción
        plt.subplot(n_images, 3, i*3 + 3)
        plt.imshow(preds_thresholded[i].squeeze(), cmap='gray')
        plt.title("Predicción U-Net")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Llamar a la función con el dataset de prueba
visualize_prediction(test_dataset, model, n_images=3)


# Iterar sobre los batches del dataset
for batch_images, batch_masks in test_dataset:
    preds = model.predict(batch_images, verbose=0)
    
    # Procesar cada imagen dentro del batch
    for i in range(len(batch_images)):
        # Configurar la fila de visualización (1 fila, 3 columnas)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. IMAGEN ORIGINAL
        # Como las imágenes están normalizadas (/255), imshow las reconoce bien
        axes[0].imshow(batch_images[i])
        axes[0].set_title(f"Imagen Original {i + 1}")
        axes[0].axis('off')

        # 2. MÁSCARA REAL (Ground Truth)
        real_mask = np.squeeze(batch_masks[i])
        axes[1].imshow(real_mask, cmap='gray')
        axes[1].set_title("Máscara Real")
        axes[1].axis('off')

        # 3. PREDICCIÓN DEL MODELO
        # Aplicamos umbral de 0.5 para que sea binaria (blanco y negro puro)
        pred = np.squeeze(preds[i])
        pred_binaria = (pred > 0.5).astype(np.uint8)
        
        axes[2].imshow(pred_binaria, cmap='gray')
        axes[2].set_title("Predicción U-Net")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    
    # Romper el bucle tras el primer batch si no quieres ver TODAS las imágenes (30)
    break 



# Realizar predicciones sobre el test dataset
for batch_images, batch_masks in test_dataset:
    preds = model.predict(batch_images)

    # Mostrar algunas de las predicciones
    for i in range(len(preds)):
        # Predicción de la imagen
        pred = np.squeeze(preds[i])  # Eliminamos el canal adicional de la predicción
        plt.imshow(pred, cmap='gray')
        plt.title(f"Predicción para imagen {i + 1}")
        plt.show()

        # Mostrar la máscara real si lo deseas
        real_mask = np.squeeze(batch_masks[i])  # La máscara real correspondiente
        plt.imshow(real_mask, cmap='gray')
        plt.title(f"Máscara Real para imagen {i + 1}")
        plt.show()


# Probamos el modelo con alguna imagen de prueba
# Ruta a la imagen de prueba
img_path = "D:/DATASETS/Imagenes/Solanaceas/Tomato/test/image-2-_JPG.rf.e46fbb9b41f9007a6337b73a038ca495.jpg"

# Cargar la imagen (en este caso, asumiendo que está en BGR, como es típico con OpenCV)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB

# Redimensionar la imagen al tamaño requerido por el modelo (en este caso, 256x256)
img_resized = cv2.resize(img, (256, 256))

# Normalizar la imagen a [0, 1] como se hace en tu dataset
img_normalized = img_resized / 255.0

# Expande la dimensión para que el modelo lo acepte (necesita un batch de imágenes, por lo que agregamos una dimensión)
img_input = np.expand_dims(img_normalized, axis=0)

# Verificar la forma de la imagen de entrada
print(f"Forma de la imagen de entrada: {img_input.shape}")

# Realizar la predicción
pred = model.predict(img_input)

# Asegurarte de que la predicción está en el rango correcto [0, 1]
pred = np.clip(pred, 0, 1)

# Mostrar la predicción de la imagen (eliminamos cualquier dimensión adicional)
plt.imshow(np.squeeze(pred), cmap='gray')
plt.title("Predicción ahora")
plt.show()

