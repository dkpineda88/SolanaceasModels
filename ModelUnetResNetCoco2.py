# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:39:24 2026

@author: dkpin
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- PARÁMETROS ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_CLASSES = 3 

class SolanaceaeMultiOutputDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ann_file, batch_size=8, img_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.batch_size = batch_size
        self.img_size = img_size
        
        # 1. Mapeo de categorías consistente
        self.cat_ids = sorted(self.coco.getCatIds()) 
        self.cat_map = {old_id: i for i, old_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids) 
        
        # Identificar dinámicamente el índice de 'Healthy' (asumiendo nombre en el JSON)
        self.healthy_idx = 1 # Valor por defecto
        for c_id, idx in self.cat_map.items():
            cat_name = self.coco.loadCats(c_id)[0]['name'].lower()
            if 'healthy' in cat_name:
                self.healthy_idx = idx
                break
                
        print(f"--- Dataset Configurado ---")
        print(f"Clases detectadas: {self.num_classes}")
        print(f"Mapa de IDs: {self.cat_map}")
        print(f"Índice detectado como Healthy: {self.healthy_idx}")
        print(f"---------------------------")

    def __len__(self):
        return len(self.ids) // self.batch_size

    def on_epoch_end(self):
        """Barajar los datos al final de cada época para mejor entrenamiento"""
        np.random.shuffle(self.ids)

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y_mask, y_class = [], [], []

        for img_id in batch_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            
            # Leer imagen
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)

            # --- PREPARACIÓN DE MÁSCARA (Lógica Softmax) ---
            # Inicializamos máscara en ceros (H, W, 4)
            mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
            
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            anns = self.coco.loadAnns(ann_ids)
            
            # Por defecto la clase es Healthy hasta que se demuestre lo contrario
            current_main_idx = self.healthy_idx 

            if anns:
                # Tomamos la categoría de la anotación más grande o la primera
                first_cat_id = anns[0]['category_id']
                current_main_idx = self.cat_map.get(first_cat_id, self.healthy_idx)

                for ann in anns:
                    idx = self.cat_map.get(ann['category_id'], -1)
                    if 0 <= idx < self.num_classes:
                        if 'segmentation' in ann and ann['segmentation']:
                            m = self.coco.annToMask(ann)
                            m_resized = cv2.resize(m, self.img_size, interpolation=cv2.INTER_NEAREST)
                            # Guardar en el canal de la enfermedad
                            mask[:, :, idx] = np.maximum(mask[:, :, idx], m_resized.astype(np.float32))

                # COMPLEMENTO SOFTMAX: 
                # Los píxeles que no pertenecen a ninguna enfermedad son 'Healthy' (Fondo)
                enfermedad_detectada = np.max(mask, axis=-1) # (256, 256) con 1s donde hay manchas
                mask[:, :, self.healthy_idx] = (enfermedad_detectada == 0).astype(np.float32)
            
            else:
                # Imagen sin anotaciones = 100% canal Healthy
                mask[:, :, self.healthy_idx] = 1.0

            # --- PREPROCESAMIENTO ---
            X.append(preprocess_input(image.astype(np.float32)))
            y_mask.append(mask)
            y_class.append(tf.keras.utils.to_categorical(current_main_idx, num_classes=self.num_classes))

        return np.array(X), {
            "mask_out": np.array(y_mask), 
            "class_out": np.array(y_class)
        }
        
def build_resnet50_unet_multi(input_shape=(256, 256, 3), num_classes=4):
    # Añade esto ANTES de compilar
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False # Congelar temporalmente
    # 1. Ajustar Skip Connections (Asegurar que las formas coincidan)
    s1 = base_model.input                                     # 256x256
    s2 = base_model.get_layer("conv1_relu").output            # 128x128
    s3 = base_model.get_layer("conv2_block3_out").output      # 64x64
    s4 = base_model.get_layer("conv3_block4_out").output      # 32x32
    
    # 2. El Bridge DEBE ser la salida final del base_model (7x7 o 8x8 aprox)
    bridge = base_model.output 

    # RAMA 1: SEGMENTACIÓN (Decoder)
    def decoder_block(inputs, skip, filters):
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
        x = layers.Resizing(skip.shape[1], skip.shape[2])(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), padding="same")(x) # Sin activación aquí
        x = layers.BatchNormalization()(x) # Crucial para el Loss de 100
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.2)(x) # Previene sobreajuste en manchas
        return x

    # Vamos subiendo desde el bridge profundo
    # Nota: ResNet50 output es 8x8 si el input es 256x256
    d1 = decoder_block(bridge, s4, 256) # 8x8 -> 32x32
    d2 = decoder_block(d1, s3, 128)     # 32x32 -> 64x64
    d3 = decoder_block(d2, s2, 64)      # 64x64 -> 128x128
    d4 = decoder_block(d3, s1, 32)      # 128x128 -> 256x256

    mask_out = layers.Conv2D(num_classes, (1, 1), activation='softmax', name="mask_out")(d4)

    # RAMA 2: CLASIFICACIÓN
    gap_deep = layers.GlobalAveragePooling2D()(bridge) # Información global
    gap_mid = layers.GlobalAveragePooling2D()(s4)      # Información de textura/manchas
    
    combined = layers.Concatenate()([gap_deep, gap_mid])
    
    fc = layers.Dense(512, activation='relu')(combined)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Dropout(0.5)(fc)
    fc = layers.Dense(128, activation='relu')(fc)
    class_out = layers.Dense(num_classes, activation='softmax', name="class_out")(fc)
    return models.Model(inputs=base_model.input, outputs=[mask_out, class_out])
# Crear y compilar
model = build_resnet50_unet_multi()

import tensorflow.keras.backend as K

def total_dice_loss(y_true, y_pred):
    # Calculamos el dice por cada canal y promediamos
    smooth = 1e-6
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    dice = K.mean((2. * intersection + smooth) / (union + smooth))
    return 1 - dice

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    # y_true y y_pred: (batch, h, w, classes)
    y_true_f = tf.cast(tf.reshape(y_true, [-1, 5]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, 5]), tf.float32)
    
    # Cálculo de Tversky Index
    # TP (True Positives)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    # FN (False Negatives) - Píxeles de enfermedad ignorados
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
    # FP (False Positives) - Ruido predicho como enfermedad
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
    
    # Alfa alto (0.7) penaliza más los FNs (sensibilidad)
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    
    # Aplicar componente Focal para ejemplos difíciles
    ft_loss = tf.pow((1 - tversky_index), 1/gamma)
    
    return tf.reduce_mean(ft_loss)

# --- COMPILACIÓN EQUILIBRADA ---
losses = {
    "mask_out": focal_tversky_loss, 
    "class_out": "categorical_crossentropy"
}

# Pesos estratégicos:
# Le damos 10 veces más importancia a la clasificación para que el modelo 
# primero aprenda qué enfermedad es, y luego dónde está.
loss_weights = {
    "mask_out": 1.0,
    "class_out": 2.0 
}

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=losses,
    loss_weights=loss_weights,
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



# Cargar Datasets
base_path = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v3i.coco/'
train_ds = SolanaceaeMultiOutputDataset(base_path +"train", base_path +"train/_annotations.coco.json")
val_ds = SolanaceaeMultiOutputDataset(base_path +"valid", base_path +"valid/_annotations.coco.json")
test_ds = SolanaceaeMultiOutputDataset(base_path +"test", base_path +"test/_annotations.coco.json")

# Entrenar
model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[early_stop, checkpoint,reduce_lr])

results = model.evaluate(test_ds)
print(f"Total Loss: {results[0]:.4f}")
print(f"Segmentación Accuracy: {results[3]*100:.2f}%")
print(f"Clasificación Accuracy: {results[4]*100:.2f}%")


import matplotlib.pyplot as plt

# Tomar un batch del dataset
X_batch, y_batch = train_ds[0]
img = X_batch[0]
mask = y_batch['mask_out'][0]
label = y_batch['class_out'][0]


plt.figure(figsize=(15, 5))

# Canal 0: Early Blight (Lo ponemos en escala de Rojos)
plt.subplot(1, 3, 1)
plt.imshow(mask[:,:,0], cmap='Reds')
plt.title(f"Canal 0: Early\nMax: {np.max(mask[:,:,0]):.2f}")

# Canal 1: Healthy (Lo ponemos en escala de Verdes)
plt.subplot(1, 3, 2)
plt.imshow(mask[:,:,1], cmap='Greens')
plt.title(f"Canal 1: Healthy\nMax: {np.max(mask[:,:,1]):.2f}")

# Canal 2: Late Blight (Lo ponemos en escala de Amarillos/Naranjas)
plt.subplot(1, 3, 3)
plt.imshow(mask[:,:,2], cmap='YlOrBr')
plt.title(f"Canal 2: Late\nMax: {np.max(mask[:,:,2]):.2f}")

plt.show()




model_version = 1
# Guardar el modelo en formato Keras (.keras)
model.save("msolUNETResNetCoco6.keras")



# 2. Configurar el conversor
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. OPTIMIZACIÓN: Evitar la cuantización si hubo pérdida
# En lugar de optimizar por tamaño, priorizamos PRECISIÓN
converter.optimizations = [] # Mantenerlo vacío para float32 puro si el tamaño no es problema
converter.target_spec.supported_types = [tf.float32]

# 4. Asegurar que las operaciones de segmentación sean compatibles
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS # Permite operaciones complejas de UNet
]

tflite_model = converter.convert()

# Guardar el nuevo modelo
with open('msolUNETResNetCoco5.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversión finalizada: Modelo en Float32 para máxima precisión.")





import numpy as np
import cv2

# 1. Extraemos la imagen del dataset (ajusta el índice si quieres otra)
img_tensor, target = test_ds[0] 
img_raw = np.array(img_tensor)

# 2. Limpieza de dimensiones (Debe quedar en [H, W, C])
if img_raw.ndim == 4: img_raw = img_raw[0] # Quitar batch si existe
if img_raw.ndim == 5: img_raw = img_raw[0, 0] # Caso del batch de 8 que vimos antes

# 3. Ajustar orden de canales si es necesario (De [C, H, W] a [H, W, C])
if img_raw.shape[0] == 3:
    img_raw = np.transpose(img_raw, (1, 2, 0))

# 4. Normalización (Imagenet) - Esto es crítico por el warning que nos dio antes
# Restamos la media y dividimos por la desviación estándar
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_normalized = (img_raw / 255.0) if img_raw.max() > 1.0 else img_raw
img_normalized = (img_normalized - mean) / std

# 5. CREACIÓN DEL INPUT FINAL (Añadimos la dimensión de batch: [1, 256, 256, 3])
input_final = np.expand_dims(img_normalized.astype(np.float32), axis=0)

print(f"Input Final creado con forma: {input_final.shape}")


import numpy as np
import matplotlib.pyplot as plt

# 1. Obtener predicción del modelo original (Keras)
# preds[0] -> Clasificación, preds[1] -> Segmentación
preds = model.predict(input_final)

mask_pred = preds[0][0]  
class_probs = preds[1][0] 

mask_data = preds[0][0] 

# 1. Separar canales
canal_early = mask_data[:, :, 0]
canal_late = mask_data[:, :, 2]
canal_fondo = mask_data[:, :, 1]

# 2. Crear una máscara combinada de enfermedad
enfermedad = canal_early + canal_late

print(f"Estadísticas de la enfermedad:")
print(f"  Max: {np.max(enfermedad):.8f}")
print(f"  Min: {np.min(enfermedad):.8f}")
print(f"  Media: {np.mean(enfermedad):.8f}")

# 3. Visualización con contraste forzado (Normalize)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(enfermedad, cmap='hot') # 'hot' ayuda a ver valores muy pequeños
plt.title(f"Mapa de Calor Forzado\nMax: {np.max(enfermedad):.4f}")
plt.colorbar()

plt.subplot(1, 3, 2)
# Solo mostramos los píxeles que están en el top 1% de confianza
umbral_top = np.max(enfermedad) * 0.8
plt.imshow(enfermedad > umbral_top, cmap='gray')
plt.title("Píxeles con 'más confianza'")

plt.subplot(1, 3, 3)
plt.imshow(canal_fondo, cmap='YlGn')
plt.title("Canal Fondo (Sano)")
plt.colorbar()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. CARGA DEL MODELO (Aquí pasas tu archivo .tflite)
model_path = "msolUNETResNetCoco3.tflite" 
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. PREPARAR LOS DATOS (input_final debe ser [1, 256, 256, 3])
# Asegúrate de usar el input_final que creamos con el padding anteriormente
interpreter.set_tensor(input_details[0]['index'], input_final)

# 3. EJECUTAR INFERENCIA
interpreter.invoke()

# 4. EXTRAER Y ANALIZAR SALIDAS
# Según tu modelo: Salida 0 = Clase, Salida 1 = Máscara
class_pred = interpreter.get_tensor(output_details[0]['index'])
mask_pred = interpreter.get_tensor(output_details[1]['index'])[0] # Quitamos el batch

print(f"--- RESULTADOS DEL MODELO ---")
print(f"Clasificación (Raw): {class_pred}")
print(f"Clase Ganadora: {np.argmax(class_pred)} (Confianza: {np.max(class_pred):.4f})")
print(f"Confianza Máxima en Segmentación: {np.max(mask_pred):.4f}")

# 5. VISUALIZACIÓN PROFUNDA DE CANALES
# Queremos ver si la segmentación existe pero tiene valores muy bajos
plt.figure(figsize=(15, 5))

# Canal 0 (Early Blight)
plt.subplot(1, 3, 1)
plt.imshow(mask_pred[:,:,0], cmap='Reds')
plt.colorbar()
plt.title(f"Canal 0: Early\nMax: {np.max(mask_pred[:,:,0]):.4f}")

# Canal 1 (Healthy / Background)
plt.subplot(1, 3, 2)
plt.imshow(mask_pred[:,:,1], cmap='Greens')
plt.colorbar()
plt.title(f"Canal 1: Healthy\nMax: {np.max(mask_pred[:,:,1]):.4f}")

# Canal 2 (Late Blight)
plt.subplot(1, 3, 3)
plt.imshow(mask_pred[:,:,2], cmap='Blues')
plt.colorbar()
plt.title(f"Canal 2: Late\nMax: {np.max(mask_pred[:,:,2]):.4f}")

plt.show()

interpreter_v2 = tf.lite.Interpreter(model_path="msolUNETResNetCoco3.tflite")
interpreter_v2.allocate_tensors()

# Usamos el input_final que creamos con normalización ImageNet
interpreter_v2.set_tensor(input_details[0]['index'], input_final)
interpreter_v2.invoke()

mask_v2 = interpreter_v2.get_tensor(output_details[1]['index'])[0]

print(f"Nueva Confianza Máxima: {np.max(mask_v2):.4f}")
# Si ahora sale > 0.50, el problema de inferencia está resuelto.