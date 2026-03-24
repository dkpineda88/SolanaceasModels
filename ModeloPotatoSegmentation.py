# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:08:47 2025

@author: dkpin
"""

import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
import numpy as np

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0

class_ids = [
    "EarlyBlight",
    "LateBlight",
    "Healthy",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Path to images and annotations
path_all = "D:/DATASETS/Imagenes/Solanaceas/PotatoXML/train/"


# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_all, file_name)
        for file_name in os.listdir(path_all)
        if file_name.endswith(".xml")
    ]
)

# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_all, file_name)
        for file_name in os.listdir(path_all)
        if file_name.endswith(".jpg")
    ]
)
print(xml_files)

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_all, image_name)
    

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)
       

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)
    

bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

# Determine the number of validation samples
num_val = int(len(xml_files) * SPLIT_RATIO)

# Split the dataset into train and validation sets
val_data = data.take(num_val)
train_data = data.skip(num_val)

import numpy as np

boxes = np.array([
    # Primera imagen
    [[0.1, 0.2, 0.4, 0.5],  # Caja 1
     [0.6, 0.1, 0.8, 0.4]], # Caja 2
    # Segunda imagen
    [[0.2, 0.3, 0.5, 0.6],  # Caja 1
     [0.4, 0.5, 0.7, 0.8],  # Caja 2
     [0.1, 0.1, 0.3, 0.3]]  # Caja 3
])

classes = np.array([
    [0, 2],    # Clases para las cajas de la primera imagen
    [1, 0, 2]  # Clases para las cajas de la segunda imagen
])

batch = 2
num_boxes = 3

bounding_boxes = {
    # num_boxes may be a Ragged dimension
    'boxes': tf.zeros(shape=[batch, num_boxes, 4]),
    'classes': tf.zeros(shape=[batch, num_boxes])
}

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.75, 1.3),
    bounding_box_format="xyxy",
)

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)

visualize_dataset(
    val_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)

yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)



class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )
        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Reset the metric state at the start of evaluation
        self.metrics.reset_state()

        # Lists to gather all boxes and classes across the batch
        all_true_boxes = []
        all_true_classes = []
        all_pred_boxes = []
        all_pred_classes = []
        all_pred_confidence = []

        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)

            # Extract true data
            y_true_boxes = y_true["boxes"]
            y_true_classes = y_true["classes"]
            y_pred_boxes = y_pred["boxes"]
            y_pred_classes = y_pred["classes"]
            y_pred_confidence = y_pred.get("confidence", None)

            # Convert numpy arrays to tensors if necessary
            if isinstance(y_true_boxes, np.ndarray):
                y_true_boxes = tf.convert_to_tensor(y_true_boxes)
            if isinstance(y_true_classes, np.ndarray):
                y_true_classes = tf.convert_to_tensor(y_true_classes)
            if isinstance(y_pred_boxes, np.ndarray):
                y_pred_boxes = tf.convert_to_tensor(y_pred_boxes)
            if isinstance(y_pred_classes, np.ndarray):
                y_pred_classes = tf.convert_to_tensor(y_pred_classes)
            if y_pred_confidence is not None and isinstance(y_pred_confidence, np.ndarray):
                y_pred_confidence = tf.convert_to_tensor(y_pred_confidence)

            # Gather all true boxes and classes
            all_true_boxes.append(y_true_boxes)
            all_true_classes.append(y_true_classes)

            # Gather all predicted boxes, classes, and confidence
            all_pred_boxes.append(y_pred_boxes)
            all_pred_classes.append(y_pred_classes)
            if y_pred_confidence is not None:
                all_pred_confidence.append(y_pred_confidence)

        # Concatenate all detections across the batch
        true_boxes_concat = tf.concat(all_true_boxes, axis=0)
        true_classes_concat = tf.concat(all_true_classes, axis=0)
        pred_boxes_concat = tf.concat(all_pred_boxes, axis=0)
        pred_classes_concat = tf.concat(all_pred_classes, axis=0)
        if all_pred_confidence:
            pred_confidence_concat = tf.concat(all_pred_confidence, axis=0)
        else:
            pred_confidence_concat = None

        # Prepare the inputs for the metric
        y_true_padded = {
            "boxes": true_boxes_concat,
            "classes": true_classes_concat,
        }
        y_pred_padded = {
            "boxes": pred_boxes_concat,
            "classes": pred_classes_concat,
        }
        if pred_confidence_concat is not None:
            y_pred_padded["confidence"] = pred_confidence_concat

        # Update the metric
        self.metrics.update_state(y_true_padded, y_pred_padded)

        # Get current metrics
        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        # Save the best model based on mAP
        current_map = metrics.get("MaP", 0)
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)

        return logs
    
yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    callbacks=[EvaluateCOCOMetricsCallback(val_ds, "model.h5")],
)

def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)    

    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


visualize_detections(yolo, dataset=val_ds, bounding_box_format="xyxy")