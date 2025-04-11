import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.preprocessing import image

# Force CPU and disable ONEDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

# --- FixedDropout that works at inference ---
class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# --- Dataset Paths ---
base_path = r".\PlantVillage"
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "val")

# --- Training Parameters ---
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

# --- Data Generators with Augmentation ---
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

val_gen = ImageDataGenerator(
    rescale=1./255,
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
    shuffle=False
)

NUM_CLASSES = len(train_generator.class_indices)

# --- Model Definition ---
conv_base = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
conv_base.trainable = True

model = Sequential([
    conv_base,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    FixedDropout(0.2),
    Dense(64, activation='relu'),
    FixedDropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[
        CategoricalAccuracy(), 
        F1Score(), 
        Precision(), 
        Recall()
    ]
)

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

# --- Evaluate Best Model ---
best_model = load_model('best_model.h5', custom_objects={'F1Score': F1Score, 'FixedDropout': FixedDropout})
val_loss, val_acc, val_f1, val_precision, val_recall = best_model.evaluate(valid_generator)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")

# --- Optional Temperature Scaled Prediction Function ---
def predict_image(img_path, model, class_names, temperature=1.5):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    logits = model.predict(img_array)[0]
    scaled_logits = logits / temperature
    softmax = tf.nn.softmax(scaled_logits).numpy()

    predicted_index = np.argmax(softmax)
    confidence = float(np.max(softmax))
    return class_names[predicted_index], confidence
