import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall

# Force CPU and disable ONEDNN for performance consistency
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import backend as K

# --- Custom F1 Score ---
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# --- Dataset Paths ---
base_path = r".\PlantVillage"
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "val")

print("Train directory:", train_dir)
print("Validation directory:", val_dir)

# --- Training Parameters ---
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

# --- Data Generators ---
train_gen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

val_gen = ImageDataGenerator(
    rescale=1/255.0,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_generator = train_gen.flow_from_directory(
    directory=train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_generator = val_gen.flow_from_directory(
    directory=val_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

NUM_CLASSES = len(train_generator.class_indices)

# --- Model Definition ---
conv_base = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
conv_base.trainable = True

model = Sequential([
    conv_base,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
    CategoricalAccuracy(), 
    F1Score(), 
    Precision(), 
    Recall()
])

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-5),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001)
]

# --- Load Previous Weights if Available ---
if os.path.exists("best_model.h5"):
    model.load_weights("best_model.h5")
    print("Loaded previous best weights.")

# --- Training ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=callbacks
)

# Optional: Save final model explicitly (if different from best_model.h5)
# model.save("plant_disease_model.h5")

# --- Load and Evaluate Best Model with Custom Metric ---
best_model = load_model('best_model.h5', custom_objects={'F1Score': F1Score})
val_loss, val_acc, val_f1, val_precision, val_recall = best_model.evaluate(valid_generator)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
