# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:03:37 2026

@author: dkpin
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#Modelo: ResNet50-U-Net (Clasificación + Segmentación)
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
        images, masks, labels = [], [], [] # Añadimos 'labels' para la clasificación
        
        target_h, target_w = self.img_size 
        
        for filename in batch_filenames:
            img_path = os.path.join(self.img_dir, filename)
            img_orig = cv2.imread(img_path)
            if img_orig is None: 
                print(f"ERROR: No se pudo cargar la imagen: {img_path}")
                continue 
            
            img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h)) 
            img = img / 255.0
            
            # --- Aquí está la clave ---
            # Asegúrate de que la máscara se inicializa a ceros
            mask = np.zeros((target_h, target_w, 3), dtype=np.float32)
           #### #mask = np.zeros((target_h, target_w), dtype=np.uint8)
            
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')
            
            # Inicializamos la clase por defecto (ej. Healthy = 1)
            class_id = 1 
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines() # Leer todas las líneas
                    if lines: # Si hay al menos una línea
                        for line in lines: # Procesar cada polígono en el archivo
                            parts = line.split()
                            if len(parts) < 9: 
                                print(f"Advertencia: Formato de línea incorrecto en {label_path}: {line.strip()}")
                                continue 
                            
                            # Obtener la clase del primer polígono (o el último si hay varios)
                            current_class_id = int(parts[0]) 
                            class_id = current_class_id # Asignar la última clase encontrada
                            
                            points = np.array(parts[1:], dtype=np.float32).reshape((4, 2))
                            
                            # ESCALADO: Multiplicar por el tamaño del lienzo
                            points[:, 0] *= target_w  # Escala X
                            points[:, 1] *= target_h  # Escala Y
                            
                            # Convertir a enteros para cv2.fillPoly
                            polygon_pts = points.astype(np.int32)
                            
                            # Pintar el polígono en la máscara
                            # ¡Asegúrate de que estás pintando con un valor visible!
                            # 2. Extraer temporalmente el canal de la clase actual (2D)
                            channel = mask[:, :, current_class_id].copy()
                            # 3. Dibujar sobre ese canal individual
                            cv2.fillPoly(channel, [polygon_pts], 1.0)
                            # 4. Asignar el canal dibujado de vuelta a la máscara original
                            mask[:, :, current_class_id] = channel
              ###              #cv2.fillPoly(mask, [polygon_pts], 1) 
                            
                            # DEBUG: Verificar si se pinta algo
                            # print(f"Pintando polígono para {filename}. Puntos: {polygon_pts}")
                            # if np.max(mask) > 0:
                            #    print(f"Máscara ya no está vacía después de fillPoly para {filename}")

            else:
                print(f"Advertencia: No se encontró archivo de etiqueta para: {filename}")
                # Si no hay label, se asume que es Healthy (class_id=1)

            images.append(img)
            masks.append(mask)
            # Convertimos class_id a One-Hot para 3 clases (0: Early, 1: Healthy, 2: Late)
            labels.append(tf.keras.utils.to_categorical(class_id, num_classes=3))

        return np.array(images), {
            "mask_out": np.array(masks),
            "class_out": np.array(labels)
        }
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_files)

# Definir parámetros (Ajusta las carpetas de 'labels')
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/'
train_dataset = YOLOOBBDataset(base_path + 'train/images/', base_path + 'train/labels/')
valid_dataset = YOLOOBBDataset(base_path + 'valid/images/', base_path + 'valid/labels/')
test_dataset = YOLOOBBDataset(base_path + 'test/images/', base_path + 'test/labels/')


import tensorflow as tf
from tensorflow.keras import layers, models


def build_resnet50_unet(input_shape=(256, 256, 3), num_classes=3):
    # 1. ENCODER: ResNet50 pre-entrenada
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Capas de salto (Skip Connections)
    s1 = base_model.input                                     # 256x256
    s2 = base_model.get_layer("conv1_relu").output           # 128x128
    s3 = base_model.get_layer("conv2_block3_out").output     # 64x64
    s4 = base_model.get_layer("conv3_block4_out").output     # 32x32
    bridge = base_model.get_layer("conv4_block6_out").output # 16x16
    
    # --- RAMA DE CLASIFICACIÓN (Con Dropout) ---
    gap = layers.GlobalAveragePooling2D()(base_model.output) 
    # Añadimos una capa densa intermedia con Dropout para regularizar la clasificación
    fc = layers.Dense(256, activation='relu')(gap)
    fc = layers.Dropout(0.5)(fc) # Apaga el 50% de las neuronas en cada paso
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(fc)

    # --- RAMA DE SEGMENTACIÓN (DECODER con Dropout) ---
    def decoder_block(inputs, skip, filters, dropout_rate=0.0):
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        
        # Aplicamos Dropout si el rate es mayor a 0
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    # Reconstrucción con Dropout progresivo
    # Usamos Dropout en las capas más profundas donde hay más parámetros
    d1 = decoder_block(bridge, s4, 256, dropout_rate=0.3) # 16x16 -> 32x32
    d2 = decoder_block(d1, s3, 128, dropout_rate=0.3)     # 32x32 -> 64x64
    d3 = decoder_block(d2, s2, 64)                       # 64x64 -> 128x128
    d4 = decoder_block(d3, s1, 32)                       # 128x128 -> 256x256
    
    # Capa final de máscara
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name="mask_out")(d4)
    
    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])


model_tomato = build_resnet50_unet()

model_tomato.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss={
        'mask_out': 'binary_crossentropy', # O 'categorical_crossentropy' si los píxeles no se solapan
        'class_out': 'categorical_crossentropy'
    },
# =============================================================================
     loss_weights={
         'mask_out': 2.0, 
         'class_out': 10.0 # Le damos prioridad a la segmentación
     },
# =============================================================================
    metrics={"mask_out": "accuracy", "class_out": "accuracy"}
)


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

history_tomato = model_tomato.fit(
    train_dataset,              # Generador de entrenamiento
    epochs=10,                  # Número de épocas
    validation_data=valid_dataset, # Generador de validación
    callbacks=[early_stop, checkpoint]
   
)

results = model_tomato.evaluate(test_dataset)
print(f"Total Loss: {results[0]:.4f}")
print(f"Segmentación Accuracy: {results[3]*100:.2f}%")
print(f"Clasificación Accuracy: {results[4]*100:.2f}%")


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

model_tomato.save("msoUNETRESNET1.keras")

# --- CONVERSIÓN OPTIMIZADA ---
# 1. Cargar el modelo desde el archivo .keras para asegurar que está limpio
model_to_convert = tf.keras.models.load_model("msoUNETRESNET1.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)

# 2. Optimizaciones estándar (reducen tamaño y mejoran compatibilidad)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] # Intentar solo Builtins primero

# 3. Evitar que Keras use opcodes muy nuevos
converter._experimental_lower_tensor_list_ops = True

try:
    tflite_model = converter.convert()
    with open("msoUNETRESNET1.tflite", "wb") as f:
        f.write(tflite_model)
    print("Modelo TFLite convertido exitosamente.")
except Exception as e:
    print(f"Error en conversión estándar: {e}")
    # Si falla, intentamos con Flex (necesita añadir dependencias en build.gradle)
    print("Reintentando con Select TF Ops...")
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open("msoUNETRESNET1.tflite", "wb") as f:
        f.write(tflite_model)
        
        
        




