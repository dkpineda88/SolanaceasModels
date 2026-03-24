# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:59:55 2026

@author: dkpin
"""


import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# 1. DATASET: Carica immagini, maschere di segmentazione e label di classificazione
class YOLOOBBDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, label_dir, batch_size=4, img_size=(256, 256), shuffle=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if self.shuffle:
            np.random.shuffle(self.img_files)
    
    def __len__(self):
        return int(np.floor(len(self.img_files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_filenames = self.img_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks, labels = [], [], []
        target_h, target_w = self.img_size 
        
        for filename in batch_filenames:
            img_path = os.path.join(self.img_dir, filename)
            img = cv2.imread(img_path)
            if img is None: continue 
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h)) / 255.0
            
            # Maschera (3 canali per 3 classi: Healthy, Early, Late)
            mask = np.zeros((target_h, target_w, 3), dtype=np.float32)
            img_class = 0 # Default (Healthy)
            
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.split()
                        if len(parts) < 9: continue 
                        class_id = int(parts[0])
                        img_class = class_id # Prende l'ultima classe trovata come label globale
                        
                        mask_val = [0, 0, 0]
                        if class_id < 3: mask_val[class_id] = 1
                        
                        points = np.array(parts[1:9], dtype=np.float32).reshape((4, 2))
                        points[:, 0] *= target_w
                        points[:, 1] *= target_h
                        cv2.fillPoly(mask, [points.astype(np.int32)], mask_val)
            
            # One-hot encoding per la classificazione
            one_hot_label = np.zeros(3, dtype=np.float32)
            if img_class < 3: one_hot_label[img_class] = 1.0
            
            images.append(img)
            masks.append(mask)
            labels.append(one_hot_label)
        
        # Converte in array numpy con tipi espliciti per evitare shape=(None,)
        return np.array(images, dtype=np.float32), {
            "mask_out": np.array(masks, dtype=np.float32),
            "classification": np.array(labels, dtype=np.float32)
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_files)
            
            


def build_mask_rcnn_with_classification(input_shape=(256, 256, 3), num_classes=3):
    # Base model (ResNet50)
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False)

    # Feature map from ResNet50
    x = base_model.output

    # --- Capa de segmentación ---
    mask_head = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name="mask_out")(x)

    # --- Capa de clasificación de la imagen completa ---
    gap = layers.GlobalAveragePooling2D()(x)  # Global Average Pooling
    class_out = layers.Dense(num_classes, activation='sigmoid', name="classification")(gap)

    # --- Modelo final ---
    model = tf.keras.models.Model(inputs=base_model.input, outputs=[mask_head, class_out])

    return model

model = build_mask_rcnn_with_classification(input_shape=(256, 256, 3), num_classes=3)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'mask_out': 'binary_crossentropy',  # Pérdida para la segmentación
        'classification': 'binary_crossentropy'  # Pérdida para la clasificación
    },
    loss_weights={
        'mask_out': 1.0,  # Peso para la segmentación
        'classification': 0.2  # Peso para la clasificación (ajustable)
    }
)




# 4. TRAINING
# Assicurati che i percorsi siano corretti nel tuo PC
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/'
train_gen = YOLOOBBDataset(base_path + 'train/images/', base_path + 'train/labels/')
valid_gen = YOLOOBBDataset(base_path + 'valid/images/', base_path + 'valid/labels/')
test_gen = YOLOOBBDataset(base_path + 'test/images/', base_path + 'test/labels/')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5),
    tf.keras.callbacks.ModelCheckpoint('mejor_modelo.keras', save_best_only=True)
]

history = model.fit(train_gen, validation_data=valid_gen, epochs=10, callbacks=callbacks)


# Evaluar el modelo
results = model.evaluate(test_gen)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")


from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

def cargar_imagen(ruta_imagen, target_size=(256, 256)):
    """
    Carga una imagen, la redimensiona y la normaliza.
    
    :param ruta_imagen: Ruta al archivo de la imagen.
    :param target_size: Tamaño al que se debe redimensionar la imagen (alto, ancho).
    :return: La imagen como un array de NumPy con la forma (1, height, width, channels).
    """
    # Cargar la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError(f"La imagen en la ruta {ruta_imagen} no se pudo cargar.")
    
    # Convertir de BGR (OpenCV) a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar la imagen
    img = cv2.resize(img, target_size)
    
    # Normalizar la imagen (esto depende de cómo esté preprocesado tu modelo)
    img = img / 255.0
    
    # Añadir la dimensión del batch
    img_array = np.expand_dims(img, axis=0)  # Forma (1, 256, 256, 3)
    
    return img_array

def predecir_imagen(model, ruta_imagen):
    """
    Realiza la predicción para una sola imagen, tanto la segmentación como la clasificación.
    
    :param model: El modelo entrenado.
    :param ruta_imagen: Ruta al archivo de la imagen.
    """
    # Cargar y preprocesar la imagen
    imagen = cargar_imagen(ruta_imagen)
    
    # Realizar la predicción
    mask_pred, class_pred = model.predict(imagen)
    
    # Convertir la máscara predicha (usamos `argmax` para obtener la clase más probable por píxel)
    mask_pred_idx = np.argmax(mask_pred[0], axis=-1)
    
    # Obtener la clasificación predicha (usamos `argmax` para obtener la clase con la mayor probabilidad)
    class_idx = np.argmax(class_pred[0])
    confianza = class_pred[0][class_idx] * 100
    
    # Mostrar los resultados
    print(f"Predicción de clasificación: {class_idx} ({confianza:.1f}%)")
    
    # Mostrar la máscara predicha
    plt.figure(figsize=(15, 5))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    img = cv2.imread(ruta_imagen)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    plt.imshow(img)
    plt.title("Imagen Original")
    plt.axis('off')
    
    # Máscara predicha
    plt.subplot(1, 3, 2)
    plt.imshow(mask_pred_idx, cmap='viridis', vmin=0, vmax=2)  # Colormap para la máscara
    plt.title("Máscara Predicha")
    plt.axis('off')
    
    # Clasificación
    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred_idx, cmap='viridis', vmin=0, vmax=2)  # Mostrar la máscara predicha
    plt.title(f"Predicción: Clase {class_idx} ({confianza:.1f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Uso del modelo para predecir una imagen
ruta_imagen = "D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/test/images/image-8-_JPG.rf.cf2a4b0e13b6a16487883d6383e6b493.jpg"
predecir_imagen(model, ruta_imagen)

def visualizar_predicciones(generator, model, num_images=4):
    # 1. Extraer un batch de datos del generador correctamente
    # generator[0] accede al __getitem__(0) que devuelve (imagenes, {targets})
    images, targets = generator[0] 
    
    # 2. Obtener las máscaras y etiquetas reales del diccionario
    masks_reales = targets["mask_out"]
    labels_reales = targets["classification"]
    
    # 3. Realizar la predicción
    # model.predict devuelve una lista: [masks_out, classification_out]
    preds = model.predict(images, verbose=0)
    print("Predicción:", preds)
    masks_predichas = preds[0]
    labels_predichos = preds[1]

    class_names = {0: "Healthy", 1: "EarlyBlight", 2: "LateBlight"}

    # Determinar cuántas imágenes mostrar (no más de las que hay en el batch)
    if images is not None and len(images) > 0:
       num_to_show = min(num_images, images.shape[0])
    else:
       print("Error: No hay imágenes para mostrar.")

    
    for i in range(num_to_show):
        plt.figure(figsize=(18, 5))

        # --- 1. IMAGEN ORIGINAL ---
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title(f"Imagen Original {i}")
        plt.axis('off')

        # --- 2. MÁSCARA REAL ---
        plt.subplot(1, 3, 2)
        # argmax convierte (256, 256, 3) -> (256, 256) seleccionando el canal con el 1
        mask_real_idx = np.argmax(masks_reales[i], axis=-1)
        plt.imshow(mask_real_idx, cmap='viridis', vmin=0, vmax=2)
        
        idx_real = np.argmax(labels_reales[i])
        plt.title(f"Diagnóstico Real: {class_names.get(idx_real, 'N/A')}")
        plt.axis('off')

        # --- 3. SEGMENTACIÓN PREDICHA ---
        plt.subplot(1, 3, 3)
        # argmax convierte probabilidades Softmax a la clase con mayor valor
        mask_pred_idx = np.argmax(masks_predichas[i], axis=-1)
        plt.imshow(mask_pred_idx, cmap='viridis', vmin=0, vmax=2)
        
        idx_pred = np.argmax(labels_predichos[i])
        confianza = labels_predichos[i][idx_pred] * 100
        plt.title(f"Predicción: {class_names.get(idx_pred, 'N/A')} ({confianza:.1f}%)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# --- Ejecución final ---
visualizar_predicciones(test_gen, model)

model_version = 1
# Guardar el modelo en formato Keras (.keras)
model.save("msolUNETFCN.keras")

import os
print(os.getcwd())  # Esto imprimirá el directorio actual de trabajo


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

open("msolUNETFCN.tflite","wb").write(tfmodel)



