import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import clone_model
from tensorflow.keras.activations import swish
from flask import jsonify
import base64
import requests

# --- Telegram Bot Config ---
TELEGRAM_BOT_TOKEN = '7844277836:AAFZau0vvjX95H2Mdg3-QYaOcymSbEHtxfc'
TELEGRAM_CHAT_ID = '1206974757'


def send_telegram_photo(photo_path, caption=''):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        response = requests.post(url, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': caption
        }, files={'photo': photo})
    return response.status_code == 200

# Custom F1 Score metric (must match what was used during training)
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
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Optional: FixedDropout class if needed
class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# Load the model with custom objects
model = load_model('best_model.h5', custom_objects={
    'F1Score': F1Score,
    'swish': swish,
    'FixedDropout': FixedDropout
})

# --- Set class names (should match your training folder structure) ---
class_names = sorted(os.listdir('./PlantVillage/train'))

# --- Predict Function ---
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    return class_names[predicted_index], confidence

# --- Evaluate on Validation Set ---
def evaluate_model():
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "./PlantVillage/val",
        image_size=(224, 224),
        batch_size=32,
        label_mode='int'
    )
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    results = model.evaluate(val_ds)
    print("\nModel Evaluation:")
    print("Loss:", results[0])
    print("Accuracy:", results[1])
    return results

# --- Save as TFLite and TensorFlow.js ---
def export_models():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,        # TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS           # Enable TensorFlow fallback ops.
    ]
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Saved TFLite model")

    os.system("tensorflowjs_converter --input_format=keras best_model.h5 web_model")
    print("✅ Saved TensorFlow.js model to ./web_model")

def send_telegram_photo(photo_path, caption=''):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        response = requests.post(url, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': caption
        }, files={'photo': photo})
    return response.status_code == 200

# --- Simple Flask App ---
app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<title>Plant Disease Predictor</title>
<h1>Upload a Plant Leaf Image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <input type=submit value=Upload>
</form>
{% if prediction %}
  <h2>Prediction: {{ prediction }}</h2>
  <h3>Confidence: {{ confidence }}%</h3>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    prediction = None
    confidence = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(path)
            prediction, confidence = predict_image(path)
            confidence = round(confidence * 100, 2)
    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence)


@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.get_json()

    if not data or 'file' not in data:
        return jsonify({'error': 'No file data'}), 400

    base64_str = data['file']
    img_data = base64.b64decode(base64_str)
    
    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', 'image.jpg')
    with open(path, 'wb') as f:
        f.write(img_data)

    prediction, confidence = predict_image(path)
    confidence_percent = round(confidence * 100, 2)

    print(f"prediction: {prediction}, confidence: {confidence_percent}")

    if confidence_percent > 90 and prediction != "Unknown":
        caption = f"🪴 Prediction: {prediction}\nConfidence: {confidence_percent}%"
        sent = send_telegram_photo(path, caption=caption)
        if sent:
            print("✅ Telegram notification sent.")
        else:
            print("❌ Failed to send Telegram notification.")

    return jsonify({
        'prediction': prediction,
        'confidence': confidence_percent
    })

# --- Entry Point ---
if __name__ == '__main__':
    #print("🧪 Evaluating model...")
    #evaluate_model()
    #print("💾 Exporting models...")
    #export_models()
    print("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0')
