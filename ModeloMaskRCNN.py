# -*- coding: utf-8 -*-
"""
Created on Thu Agosto  25 11:08:47 2025

@author: dkpin
"""

import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage.draw
import random

ROOT_DIR = 'D:/Articulos/Art.Segmentacion/Modelos/MaskRCNN_Video-master'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'

sys.path.append(ROOT_DIR) 

import tensorflow as tf
from tensorflow import keras


from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class CustomConfig(Config):
    """Configuration for training on the dataset."""
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3  # background + Healthy + LateBlight + EarlyBlight
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 5
    BACKBONE = 'resnet50'
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = CustomConfig()
config.display()

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset in COCO format."""
        self.add_class("object", 1, "Healthy")
        self.add_class("object", 2, "LateBlight")
        self.add_class("object", 3, "EarlyBlight")

        assert subset in ["train", "valid"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations_file = os.path.join(dataset_dir, "_annotations.coco.json")
        with open(annotations_file) as f:
            annotations_data = json.load(f)

        annotations = annotations_data['annotations']
        images = annotations_data['images']
        categories = annotations_data['categories']
        
        category_map = {category['id']: category['name'] for category in categories}

        for image in images:
            image_id = image['id']
            image_path = os.path.join(dataset_dir, image['file_name'])
            image_width, image_height = image['width'], image['height']

            image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

            polygons = []
            class_ids = []
            for ann in image_annotations:
                x_min, y_min, width, height = ann['bbox']
                polygons.append({
                    'all_points_x': [x_min, x_min + width, x_min + width, x_min],
                    'all_points_y': [y_min, y_min, y_min + height, y_min + height]
                })
                class_ids.append(ann['category_id'])

            self.add_image("object", 
                           image_id=image_id,
                           path=image_path,
                           width=image_width, 
                           height=image_height,
                           polygons=polygons,
                           class_ids=class_ids)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        polygons = image_info['polygons']
        class_ids = image_info['class_ids']
        height, width = image_info['height'], image_info['width']

        mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

        for i, p in enumerate(polygons):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask, np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        image_info = self.image_info[image_id]
        return image_info["path"]

# Create the dataset object
dataset_train = CustomDataset()
dataset_train.load_custom("D:/DATASETS/Imagenes/Solanaceas/Tomato", "train")
dataset_train.prepare()

# Create the validation dataset (if needed)
dataset_val = CustomDataset()
dataset_val.load_custom("D:/DATASETS/Imagenes/Solanaceas/Tomato", "valid")
dataset_val.prepare()



# Habilitar la ejecución ansiosa
tf.compat.v1.enable_eager_execution()



# Create model in training mode
# Crear el modelo en modo de entrenamiento
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Cargar pesos COCO, excluyendo las capas de salida que no coinciden
# Cargar pesos COCO, excluyendo las capas específicas
# Cargar pesos COCO, excluyendo las capas de clases y cajas
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"])

# Reconfigurar las capas de salida (ajustar el número de clases)
model.keras_model.get_layer('mrcnn_class_logits').output = tf.keras.layers.Dense(3, activation='softmax')(model.keras_model.get_layer('mrcnn_class_logits').output)  # Ajusta 3 clases
model.keras_model.get_layer('mrcnn_bbox_fc').output = tf.keras.layers.Dense(3)(model.keras_model.get_layer('mrcnn_bbox_fc').output)  # Ajusta 3 cajas por clase

# Vuelve a compilar el modelo después de modificar las capas
model.keras_model.compile(optimizer=tf.keras.optimizers.Adam(lr=config.LEARNING_RATE), loss={'mrcnn_class_loss': 'categorical_crossentropy', 'mrcnn_bbox_loss': 'mse'})

# Ahora puedes continuar con el entrenamiento
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='heads')
