# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 22:25:18 2026

@author: dkpin
"""

# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- 1. DATASET CON FILTRO DE SEGURIDAD (9 CLASES) ---
class YOLOOBBDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, label_dir, batch_size=4, img_size=(256, 256), shuffle=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = 9  # Definido para tus 9 clases de tomate
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

            mask = np.zeros((target_h, target_w, self.num_classes), dtype=np.float32)
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')

            # Clase por defecto: Healthy (ID 2)
            class_id = 2 
            
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # VALIDACIÓN DE ID: Si el TXT trae un ID >= 9 (como el 15 que te dio error)
                        raw_id = int(lines[0].split()[0])
                        class_id = raw_id if raw_id < self.num_classes else 2
                        
                        for line in lines:
                            parts = line.split()
                            if len(parts) < 9: continue
                            
                            curr_class = int(parts[0])
                            # Solo dibujamos si la clase está en el rango 0-8
                            if 0 <= curr_class < self.num_classes:
                                points = np.array(parts[1:9], dtype=np.float32).reshape((4, 2))
                                points[:, 0] *= target_w
                                points[:, 1] *= target_h
                                
                                temp_channel = np.zeros((target_h, target_w), dtype=np.float32)
                                cv2.fillPoly(temp_channel, [points.astype(np.int32)], 1.0)
                                mask[:, :, curr_class] = np.maximum(mask[:, :, curr_class], temp_channel)
            else:
                mask[:, :, 2] = 1.0 # Si no hay label, toda la máscara es "Healthy"

            images.append(img)
            masks.append(mask)
            labels.append(tf.keras.utils.to_categorical(class_id, num_classes=self.num_classes))

        return np.array(images, dtype=np.float32), (
            np.array(masks, dtype=np.float32), 
            np.array(labels, dtype=np.float32)
        )

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.img_files)

# --- 2. MODELO UNET DINÁMICO ---

def build_unet_mobilenet_9classes(input_shape=(256, 256, 3), num_classes=9):
    inputs = layers.Input(shape=input_shape) 
    

    # 1. ENCODER: MobileNetV3Large
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=input_shape, 
        include_top=False, 
        weights='imagenet', 
        input_tensor=inputs
    )

    # Búsqueda dinámica de capas para Skip Connections
    skip_layers = {}
    for layer in base_model.layers:
        try:
            shape = layer.output.shape
            if shape[1] in [128, 64, 32]:
                skip_layers[shape[1]] = layer.output
        except: continue

    bridge = base_model.output 

    # 2. DECODER (U-Net)
    def up_block(x, skip, filters, size):
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Resizing(size, size)(x) 
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x) # Mantiene los gradientes controlados
        return x

    # Reconstrucción (Subida)
    x_dec = up_block(bridge, skip_layers[32], 128, 32)
    x_dec = up_block(x_dec, skip_layers[64], 64, 64)
    x_dec = up_block(x_dec, skip_layers[128], 32, 128)

    x_dec = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x_dec)
    x_dec = layers.Resizing(256, 256)(x_dec)

    # SALIDA 1: SEGMENTACIÓN
    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='mask_out')(x_dec)

    # 3. CABEZA DE CLASIFICACIÓN (Optimizada)
    # Agregamos un Flatten o GlobalPool pero con una capa más densa para procesar las 9 clases
    y = layers.GlobalAveragePooling2D()(bridge)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y) # Un poco más de Dropout para generalizar mejor
    y = layers.Dense(256, activation='relu')(y)
    y = layers.Dropout(0.3)(y)
    
    # SALIDA 2: CLASIFICACIÓN
    class_out = layers.Dense(num_classes, activation='softmax', name='class_out')(y)

    model = models.Model(inputs=inputs, outputs=[mask_out, class_out])
    return model

# --- INSTANCIACIÓN Y COMPILACIÓN ---
tf.keras.backend.clear_session()


import tensorflow.keras.backend as K

# Definir la función antes
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1e-6) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1e-6)

model_9 = build_unet_mobilenet_9classes()

# Descongelar las últimas capas del encoder para especializarlo
# 1. Primero, asegúrate de que todo sea entrenable
for layer in model_9.layers:
    layer.trainable = True

# 2. Bloqueamos las primeras 100 capas (bordes, colores, formas simples)
# Esto protege los pesos de ImageNet que ya funcionan bien.
for layer in model_9.layers[:100]:
    layer.trainable = False

# 3. Verificación
print(f"Capa {model_9.layers[99].name} está bloqueada: {not model_9.layers[99].trainable}")
print(f"Capa {model_9.layers[150].name} está abierta para aprender: {model_9.layers[150].trainable}")

model_9.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6), # 5 veces más rápido que antes, pero aún seguro
    loss={
        "mask_out": dice_loss, 
        "class_out": "categorical_crossentropy"
    },
    loss_weights={
        "mask_out": 2.0,  # Subimos un poco para que no descuide la forma de la mancha
        "class_out": 10.0   # Mantenemos prioridad en clasificación
    },
    metrics={"mask_out": "accuracy", "class_out": "accuracy"}
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint('mejor_modelo_9clases.keras', save_best_only=True),
    ReduceLROnPlateau(
        monitor='val_class_out_accuracy', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-8, 
        verbose=1)
    
]


# Agrégalo a tu lista de callbacks en el model.fit
# callbacks=[checkpoint, reduce_lr, early_stop]

# --- 3. COMPILACIÓN Y ENTRENAMIENTO ---
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDiseaseYolov8-obb/'
train_ds = YOLOOBBDataset(base_path + 'train/images/', base_path + 'train/labels/')
valid_ds = YOLOOBBDataset(base_path + 'valid/images/', base_path + 'valid/labels/')
test_ds = YOLOOBBDataset(base_path + 'test/images/', base_path + 'test/labels/')

# 3. Entrenar
history = model_9.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=30,
    callbacks=callbacks
)


results = model_9.evaluate(test_ds)
print(f"Total Loss: {results[0]:.4f}")
print(f"Segmentación Accuracy: {results[3]*100:.2f}%")
print(f"Clasificación Accuracy: {results[4]*100:.2f}%")

# Toma una imagen del test
imgs, (msks, lbls) = test_ds[0]
preds = model_9.predict(imgs)

# Visualiza la primera imagen del batch
# En lugar de argmax, suma los canales de las 9 enfermedades




plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(imgs[0])

plt.subplot(1, 3, 2)
plt.title("Suma de Máscaras Reales")
plt.imshow(np.sum(msks[0], axis=-1), cmap='gray') # Debería verse la hoja en blanco y negro


plt.subplot(1, 3, 3)
plt.title("Confianza de Predicción")
plt.imshow(np.max(preds[0][0], axis=-1), cmap='hot') # Muestra dónde el modelo "cree" que hay algo
plt.show()

img_batch, (mask_batch, label_batch) = test_ds[0]

print(f"Máscara - Valor Máximo: {np.max(mask_batch)}") # DEBE SER 1.0
print(f"Máscara - Forma: {mask_batch.shape}")        # DEBE SER (1, 256, 256, 9)

plt.imshow(np.sum(mask_batch[0], axis=-1), cmap='gray')
plt.title("¿Ves algo blanco? Si es negro, el error es el Dataset")
plt.show()

# 1. Obtener un batch
img_batch, (mask_batch, label_batch) = train_ds[0]

# 2. Buscar en qué píxeles hay datos (valor 1.0)
indices_con_datos = np.where(mask_batch > 0)

if len(indices_con_datos[0]) > 0:
    print(f"✅ ¡Éxito! Se encontraron {len(indices_con_datos[0])} píxeles con etiquetas.")
    print(f"Ejemplo de coordenadas con manchas: {indices_con_datos[1][0]}, {indices_con_datos[2][0]}")
    
    # 3. Visualización Gigante
    plt.figure(figsize=(10, 10))
    # Sumamos todos los canales y todas las imágenes del batch para forzar que algo aparezca
    visual_mask = np.sum(mask_batch, axis=(0, -1)) 
    plt.imshow(visual_mask, cmap='hot')
    plt.colorbar()
    plt.title("Mapa de calor de las manchas detectadas")
    plt.show()
else:
    print("❌ Sigue saliendo negro. Revisa si los archivos .txt tienen las coordenadas correctas.")

###MATRIZ DE CONFUSION
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_labels = []
all_preds = []

for i in range(len(test_ds)):
    imgs, (msks, lbls) = test_ds[i]
    p_mask, p_class = model_9.predict(imgs, verbose=0)
    all_labels.extend(np.argmax(lbls, axis=1))
    all_preds.extend(np.argmax(p_class, axis=1))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()