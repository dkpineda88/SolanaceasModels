# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 20:56:34 2026

@author: dkpin
"""

# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 1. DATASET CLASS CORREGIDA ---
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
            img_orig = cv2.imread(img_path)
            if img_orig is None: continue 
            
            img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h)) / 255.0
            
            # Inicializamos máscara de 3 canales (0: Early, 1: Healthy, 2: Late)
            mask = np.zeros((target_h, target_w, 3), dtype=np.float32)
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')
            
            class_id = 1 # Por defecto Healthy
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        class_id = int(lines[0].split()[0])
                        for line in lines:
                            parts = line.split()
                            if len(parts) < 9: continue
                            
                            curr_id = int(parts[0])
                            # YOLO OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalizados)
                            points = np.array(parts[1:9], dtype=np.float32).reshape((4, 2))
                            points[:, 0] *= target_w
                            points[:, 1] *= target_h
                            
                            # Dibujar en el canal correspondiente
                            channel = np.zeros((target_h, target_w), dtype=np.float32)
                            cv2.fillPoly(channel, [points.astype(np.int32)], 1.0)
                            mask[:, :, curr_id] = np.maximum(mask[:, :, curr_id], channel)
            else:
                # Si es healthy, toda la hoja (o la máscara) podría ser 1.0 en el canal 1
                mask[:, :, 1] = 1.0

            images.append(img)
            masks.append(mask)
            labels.append(tf.keras.utils.to_categorical(class_id, num_classes=3))

        return np.array(images, dtype=np.float32), (
            np.array(masks, dtype=np.float32), 
            np.array(labels, dtype=np.float32)
        )

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.img_files)

# --- 2. MODELO DINÁMICO ---
def build_unet_mobilenet(input_shape=(256, 256, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=input_shape, include_top=False, weights='imagenet', input_tensor=inputs
    )

    skip_layers = {}
    for layer in base_model.layers:
        try:
            shape = layer.output.shape
            if shape[1] in [128, 64, 32]: skip_layers[shape[1]] = layer.output
        except: continue

    bridge = base_model.output 
    
    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(bridge)
    x = layers.Resizing(32, 32)(x)
    x = layers.Concatenate()([x, skip_layers[32]])
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Resizing(64, 64)(x)
    x = layers.Concatenate()([x, skip_layers[64]])
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Resizing(128, 128)(x)
    x = layers.Concatenate()([x, skip_layers[128]])
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Resizing(256, 256)(x)

    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='mask_out')(x)
    y = layers.GlobalAveragePooling2D()(bridge)
    class_out = layers.Dense(num_classes, activation='softmax', name='class_out')(y)

    return models.Model(inputs=inputs, outputs=[mask_out, class_out])

# --- 3. FLUJO DE EJECUCIÓN ---
tf.keras.backend.clear_session()
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/'
train_dataset = YOLOOBBDataset(base_path + 'train/images/', base_path + 'train/labels/')
valid_dataset = YOLOOBBDataset(base_path + 'valid/images/', base_path + 'valid/labels/')
test_dataset = YOLOOBBDataset(base_path + 'test/images/', base_path + 'test/labels/')



model_tomato = build_unet_mobilenet(num_classes=3)

def iou(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2))
    union = tf.reduce_sum(y_true, axis=(1, 2)) + tf.reduce_sum(y_pred, axis=(1, 2)) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))

model_tomato.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss={"mask_out": "categorical_crossentropy", "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 0.5, "class_out": 10.0},
    metrics={"mask_out": iou, "class_out": "accuracy"}
)


# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
checkpoint = ModelCheckpoint('mejor_modelo_tomate.keras', save_best_only=True)

class_weights = {0: 2.0, 1: 1.0, 2: 0.5}  # Adjust weights according to your dataset distribution


# Entrenamiento
history = model_tomato.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10, # Aumenta según necesites
    #class_weight=class_weights,  # Add class weights
    callbacks=[early_stop, checkpoint]
)


results = model_tomato.evaluate(test_dataset)
print(f"Total Loss: {results[0]:.4f}")
print(f"Segmentación Accuracy: {results[3]*100:.2f}%")
print(f"Clasificación Accuracy: {results[4]*100:.2f}%")


# Después del primer fit...
model_tomato.trainable = True
model_tomato.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Muy bajo
    loss={"mask_out": "binary_crossentropy", "class_out": "categorical_crossentropy"},
    loss_weights={"mask_out": 5.0, "class_out": 1.0},
    metrics={"class_out": "accuracy"}
)
# Entrenar otras 5-10 épocas

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

model_tomato.save("modeloTomateUnetMobilnet.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model_tomato)

# ESTA ES LA LÍNEA CLAVE: Forzar compatibilidad con versiones anteriores
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Operaciones estándar de TFLite
    tf.lite.OpsSet.SELECT_TF_OPS    # Permite operaciones de TF original si es necesario
]

# Opcional: Desactiva optimizaciones experimentales que suben la versión de los opcodes
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()


with open('modeloTomateUnetMobilnet.tflite', 'wb') as f:
    f.write(tflite_model)

print("¡Modelo optimizado con éxito!")
