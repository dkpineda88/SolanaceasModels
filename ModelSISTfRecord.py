import tensorflow as tf
import keras
from keras import ops, layers
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (256, 256)
NUM_CLASSES = 4  # [Fondo, EarlyBlight, Healthy, LateBlight]
BATCH_SIZE = 8
EPOCHS = 25

# Rutas de tus archivos (Ajusta los nombres según tu descarga)
TRAIN_TFRECORD = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v3i.tfrecord/train/EarlyBlight.tfrecord'
VAL_TFRECORD = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v3i.tfrecord/valid/EarlyBlight.tfrecord'
TEST_TFRECORD = 'D:/DATASETS/Imagenes/Solanaceas/TomatoDisease.v3i.tfrecord/test/EarlyBlight.tfrecord'

# --- 1. CAPAS PERSONALIZADAS (SIS-YOLOv8) ---

class StyleRandomization(layers.Layer):
    """Capa para inyectar robustez ante cambios de iluminación (SIS Method)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        # Usamos ops de Keras para compatibilidad
        mean = ops.mean(x, axis=[1, 2], keepdims=True)
        variance = ops.var(x, axis=[1, 2], keepdims=True)
        std = ops.sqrt(variance + 1e-5)
        x_norm = (x - mean) / std
        
        if training:
            # Perturbación aleatoria solo en entrenamiento
            shape = ops.shape(mean)
            gamma = tf.random.uniform(shape, 0.9, 1.1)
            beta = tf.random.uniform(shape, -0.1, 0.1)
            return x_norm * gamma + beta
        return x_norm

# --- 2. MÓDULOS DE CONSTRUCCIÓN ---

def fusion_inception_module(x, filters):
    b1 = layers.Conv2D(filters // 4, (1, 1), padding='same', activation='swish')(x)
    b2 = layers.Conv2D(filters // 4, (3, 3), padding='same', activation='swish')(x)
    b3 = layers.Conv2D(filters // 4, (5, 5), padding='same', activation='swish')(x)
    b4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    b4 = layers.Conv2D(filters // 4, (1, 1), padding='same', activation='swish')(b4)
    return layers.Concatenate()([b1, b2, b3, b4])

def c2f_sis_block(x, filters):
    x = StyleRandomization()(x)
    c1 = layers.Conv2D(filters, (1, 1), activation='swish')(x)
    # Usamos keras.ops.split en lugar de tf.split
    split = ops.split(c1, 2, axis=-1)
    out = layers.Add()([split[0], split[1]])
    return layers.Conv2D(filters, (1, 1), activation='swish')(out)

def sppf_is_module(x, filters):
    c1 = layers.Conv2D(filters // 2, (1, 1), activation='swish')(x)
    p1 = layers.MaxPooling2D(5, strides=1, padding='same')(c1)
    p2 = layers.MaxPooling2D(9, strides=1, padding='same')(p1)
    p3 = layers.MaxPooling2D(13, strides=1, padding='same')(p2)
    concat = layers.Concatenate()([c1, p1, p2, p3])
    return layers.Conv2D(filters, (1, 1), activation='swish')(concat)


# --- 3. PARSEO DE DATOS ---
def parse_proto(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/mask': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed['image/encoded'], channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0 # Normalización directa para SIS
    
    masks = tf.sparse.to_dense(parsed['image/object/mask'], default_value='')
    labels = tf.sparse.to_dense(parsed['image/object/class/label'], default_value=0)
    
    final_mask = tf.zeros((*IMG_SIZE, NUM_CLASSES), dtype=tf.float32)
    main_label = tf.cast(labels[0], tf.int32) if tf.shape(labels)[0] > 0 else 2

    def process_masks():
        m_acc = tf.zeros((*IMG_SIZE, NUM_CLASSES), dtype=tf.float32)
        for i in range(tf.shape(masks)[0]):
            m = tf.io.decode_png(masks[i], channels=1)
            m = tf.image.resize(m, IMG_SIZE, method="nearest")
            m_bin = tf.cast(m > 0, tf.float32)
            l_idx = tf.cast(labels[i], tf.int32)
            m_acc = tf.maximum(m_acc, m_bin * tf.one_hot(l_idx, NUM_CLASSES))
        return m_acc

    final_mask = tf.cond(tf.shape(masks)[0] > 0, process_masks, lambda: final_mask)
    return image, {"mask_out": final_mask, "class_out": tf.one_hot(main_label, NUM_CLASSES)}

def get_dataset(path):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = get_dataset(TRAIN_TFRECORD)
val_ds = get_dataset(VAL_TFRECORD)
test_ds = get_dataset(TEST_TFRECORD)

# --- 3. CONSTRUCCIÓN DEL MODELO ---

def build_sis_model():
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    
    # Encoder
    x1 = fusion_inception_module(inputs, 32) 
    x2 = layers.Conv2D(64, (3, 3), strides=2, padding='same')(x1) 
    x2 = c2f_sis_block(x2, 64)
    
    x3 = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x2) 
    x3 = c2f_sis_block(x3, 128)
    
    x4 = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x3) 
    x4 = sppf_is_module(x4, 256)
    
    # Rama Clasificación
    gap = layers.GlobalAveragePooling2D()(x4)
    class_out = layers.Dense(NUM_CLASSES, activation='softmax', name='class_out')(gap)
    
    # Decoder (Segmentación)
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(x4)
    u1 = layers.Concatenate()([u1, x3])
    
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(u1)
    u2 = layers.Concatenate()([u2, x2])
    
    u3 = layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(u2)
    u3 = layers.Concatenate()([u3, x1])
    
    mask_out = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='mask_out')(u3)
    
    return keras.Model(inputs=inputs, outputs=[mask_out, class_out])



def aspp_block(x, filters):
    """Módulo de la Etapa 3 para segmentación fina de manchas"""
    shape = ops.shape(x)
    
    # 1. Conv 1x1
    b0 = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation('swish')(b0)
    
    # 2. Atrous Convolutions (Dilation rates: 6, 12, 18)
    # Esto busca manchas pequeñas y grandes simultáneamente
    atrous_rates = [6, 12, 18]
    branches = [b0]
    
    for rate in atrous_rates:
        b = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=rate, use_bias=False)(x)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('swish')(b)
        branches.append(b)
        
    # 3. Image Pooling (Contexto global de la hoja)
    pool = layers.GlobalAveragePooling2D()(x)
    pool = layers.Reshape((1, 1, ops.shape(x)[-1]))(pool)
    pool = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(pool)
    pool = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation='bilinear')(pool)
    branches.append(pool)
    
    # Fusión de todas las escalas
    concat = layers.Concatenate()(branches)
    out = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(concat)
    return layers.Activation('swish')(out)

def build_triple_stage_model():
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    
    # --- ETAPA 1: BACKBONE (Aislamiento de Hoja) ---
    x1 = fusion_inception_module(inputs, 32)
    x2 = layers.Conv2D(64, (3, 3), strides=2, padding='same')(x1)
    x2 = c2f_sis_block(x2, 64)
    
    x3 = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x2)
    x3 = c2f_sis_block(x3, 128)
    
    x4 = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x3)
    
    # --- ETAPA 2: CLASIFICACIÓN (Identificación) ---
    gap = layers.GlobalAveragePooling2D()(x4)
    class_out = layers.Dense(NUM_CLASSES, activation='softmax', name='class_out')(gap)
    
    # --- ETAPA 3: SEGMENTACIÓN SEMÁNTICA (Localización fina) ---
    # Aplicamos ASPP en el punto de mayor abstracción
    aspp = aspp_block(x4, 256)
    
    # Decoder con Skip Connections (U-Net style)
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(aspp)
    u1 = layers.Concatenate()([u1, x3])
    u1 = layers.Conv2D(128, (3, 3), padding='same', activation='swish')(u1)
    
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(u1)
    u2 = layers.Concatenate()([u2, x2])
    
    u3 = layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(u2)
    u3 = layers.Concatenate()([u3, x1])
    
    # Salida final de píxeles
    mask_out = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='mask_out')(u3)
    
    return keras.Model(inputs=inputs, outputs=[mask_out, class_out])

model = build_triple_stage_model()

# --- 5. COMPILACIÓN Y ENTRENAMIENTO ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={'mask_out': 'categorical_crossentropy', 'class_out': 'categorical_crossentropy'},
    loss_weights={'mask_out': 20.0, 'class_out': 1.0},
    metrics={"mask_out": "accuracy", "class_out": "accuracy"}
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint('sis_yolov8_tomato.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_mask_out_loss', factor=0.5, patience=3)
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

results = model.evaluate(test_ds)
print(f"Total Loss: {results[0]:.4f}")
print(f"Segmentación Accuracy: {results[3]*100:.2f}%")
print(f"Clasificación Accuracy: {results[4]*100:.2f}%")



# --- 6. VISUALIZACIÓN ---
def visualize_results(dataset, model):
    for images, targets in dataset.take(1):
        preds = model.predict(images)
        for i in range(3):
            plt.figure(figsize=(12, 4))
            # Imagen
            plt.subplot(1, 3, 1); plt.imshow(images[i]); plt.title("Original")
            # Real
            gt = np.max(targets['mask_out'][i].numpy()[..., 1:], axis=-1)
            plt.subplot(1, 3, 2); plt.imshow(gt, cmap='gray'); plt.title("Real Mask")
            # Pred
            pr = np.max(preds[0][i][..., 1:], axis=-1)
            p_class = np.argmax(preds[1][i])
            plt.subplot(1, 3, 3); plt.imshow(pr, cmap='hot'); plt.title(f"Pred: {p_class}")
            plt.show()

visualize_results(val_ds, model)