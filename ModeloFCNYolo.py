# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:11:13 2026

@author: dkpin
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, applications


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
            
            print(type(label_path))
            print(label_path)
            
            if os.path.exists(label_path.decode('utf-8')):
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


def get_mask_from_obb(label_path, img_size=(256, 256)):
    """Convierte el archivo .txt de YOLO OBB en una máscara binaria."""
    h_target, w_target = img_size
    mask = np.zeros((h_target, w_target, 1), dtype=np.float32)

    # Si label_path es un Tensor de tipo bytes, lo decodificamos a string
    if isinstance(label_path, tf.Tensor):
        label_path = label_path.numpy()  # Convertir el tensor a un array de NumPy
    if isinstance(label_path, bytes):  # Si es bytes, decodificamos
        label_path = label_path.decode('utf-8')
        
        
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                # Obtener puntos y desnormalizar
                points = np.array(parts[1:], dtype=np.float32).reshape((4, 2))
                points[:, 0] *= w_target  # x
                points[:, 1] *= h_target  # y
                
                # Dibujar el polígono en la máscara
                cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
    return mask



def process_path(img_path, label_path):
    """Función de preprocesamiento para tf.data"""
    # Cargar Imagen
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256]) / 255.0

    # Cargar Máscara (usamos py_function porque cv2 no es nativo de TF)
    [mask] = tf.py_function(get_mask_from_obb, [label_path], [tf.float32])
    mask.set_shape([256, 256, 1])
    
    return img, mask


def create_dataset(base_dir, subset, batch_size=4):
    img_dir = os.path.join(base_dir, subset, 'images')
    label_dir = os.path.join(base_dir, subset, 'labels')
    
    # Listar archivos
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    label_files = sorted([os.path.join(label_dir, os.path.splitext(f)[0] + '.txt') 
                         for f in os.listdir(img_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((img_files, label_files))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Carga de tus datos
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/'

train_ds = create_dataset(base_path, 'train')
valid_ds = create_dataset(base_path, 'valid')


def FCN_Binary(input_shape=(256, 256, 3)):
    """
    Define una arquitectura FCN-32 para segmentación binaria.
    """
    # 1. Codificador (Backbone preentrenado)
    # Usamos VGG16 sin las capas densas finales
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # 2. Capas Convolucionales intermedias (Sustituyen a las Fully Connected)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(base_model.output)
    x = layers.Dropout(0.3)(x)
    
    # 3. Capa de Clasificación 1x1
    # filters=1 porque es segmentación binaria (fondo vs objeto)
    # kernel_size=(1,1) para predecir por cada pixel
    score = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(x)
    
    # 4. Decodificador (Upsampling / Deconvolución)
    # VGG16 reduce la imagen 5 veces (2^5 = 32), por lo que aplicamos un strides de 32
    # para recuperar el tamaño original (de 8x8 a 256x256)
    outputs = layers.Conv2DTranspose(
        filters=1, 
        kernel_size=(64, 64), 
        strides=(32, 32), 
        padding='same', 
        activation='sigmoid'
    )(score)

    # Definir el modelo final
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model

# --- INSTANCIACIÓN DEL MODELO ---
model = FCN_Binary(input_shape=(256, 256, 3))
model.summary()


# Importante: Asegura que el modelo termine con 1 canal y activación 'sigmoid'
model = FCN_Binary(input_shape=(256, 256, 3)) 

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
)


# El entrenamiento es más fluido con tf.data
model.fit(train_ds, validation_data=valid_ds, epochs=25)

results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

