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



# 2. ARCHITETTURA: U-Net Multi-Output
import tensorflow as tf
from tensorflow.keras import layers

def build_unet_with_classification(input_shape=(256, 256, 3), num_classes=3):
    inputs = tf.keras.layers.Input(input_shape, name="main_input")

    # --- Encoder (contracción) ---
    s1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    s1 = layers.BatchNormalization()(s1)
    s1 = layers.Conv2D(64, 3, activation="relu", padding="same")(s1)
    s1 = layers.BatchNormalization()(s1)
    p1 = layers.MaxPooling2D((2, 2))(s1)

    s2 = layers.Conv2D(128, 3, activation="relu", padding="same")(p1)
    s2 = layers.BatchNormalization()(s2)
    s2 = layers.Conv2D(128, 3, activation="relu", padding="same")(s2)
    s2 = layers.BatchNormalization()(s2)
    p2 = layers.MaxPooling2D((2, 2))(s2)

    s3 = layers.Conv2D(256, 3, activation="relu", padding="same")(p2)
    s3 = layers.BatchNormalization()(s3)
    s3 = layers.Conv2D(256, 3, activation="relu", padding="same")(s3)
    s3 = layers.BatchNormalization()(s3)
    p3 = layers.MaxPooling2D((2, 2))(s3)

    # Bottleneck
    b1 = layers.Conv2D(512, 3, activation="relu", padding="same")(p3)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Conv2D(512, 3, activation="relu", padding="same")(b1)
    b1 = layers.BatchNormalization()(b1)

    # --- Decoder (expansión) con skip connections ---
    u4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(b1)
    u4 = layers.concatenate([u4, s3])  # Skip connection
    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(u4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(c4)
    c4 = layers.BatchNormalization()(c4)

    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = layers.concatenate([u5, s2])  # Skip connection
    c5 = layers.Conv2D(128, 3, activation="relu", padding="same")(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(128, 3, activation="relu", padding="same")(c5)
    c5 = layers.BatchNormalization()(c5)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, s1])  # Skip connection
    c6 = layers.Conv2D(64, 3, activation="relu", padding="same")(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(64, 3, activation="relu", padding="same")(c6)
    c6 = layers.BatchNormalization()(c6)

    # --- Salida de la máscara de segmentación ---
    mask_out = layers.Conv2D(num_classes, (1, 1), activation="softmax", name="mask_out")(c6)

    # --- Salida de clasificación ---
    gap = layers.GlobalAveragePooling2D()(b1)  # Global Average Pooling
    class_out = layers.Dense(num_classes, activation="sigmoid", name="classification")(gap)

    # Modelo
    model = tf.keras.models.Model(inputs, [mask_out, class_out])

    return model



# 3. SETUP E COMPILAZIONE
model = build_unet_with_classification()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.1),
])
# Úsalo así: x = data_augmentation(inputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'mask_out': 'binary_crossentropy',  # Para segmentación binaria por clase
        'classification': 'binary_crossentropy'  # Clasificación multietiqueta
    },
    loss_weights={
        'mask_out': 1.0,
        'classification': 0.2  # Ajusta según la importancia de cada tarea
    }
)



# 4. TRAINING
# Assicurati che i percorsi siano corretti nel tuo PC
base_path = 'D:/DATASETS/Imagenes/Solanaceas/PotatoDisease.v3i.yolov8-obb/'
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


# Verificar la salida de la predicción


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


  
    

model_version = 1
# Guardar el modelo en formato Keras (.keras)
model.save("msolUNET2_2_potato.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ESTA ES LA LÍNEA CLAVE: Forzar compatibilidad con versiones anteriores
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Operaciones estándar de TFLite
    tf.lite.OpsSet.SELECT_TF_OPS    # Permite operaciones de TF original si es necesario
]

# Opcional: Desactiva optimizaciones experimentales que suben la versión de los opcodes
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

with open("msolUNET2_2_potato.tflite", "wb") as f:
    f.write(tflite_model)
    
open("msolUNET2_2_potato.tflite","wb").write(tflite_model)







