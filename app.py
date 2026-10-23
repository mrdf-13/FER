from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

# ---------------- CONFIG -----------------
MODEL_PATH = "emotiondetector.h5"           # your trained model file
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # modify for your model
INPUT_SHAPE = (48, 48)            # modify to match model input shape
# -----------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
model = load_model(MODEL_PATH)

# Helper: check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: preprocess image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=INPUT_SHAPE, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Routes
@app.route('/')
def home():
    return render_template("result.html", result=None, image_url=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("result.html", result="No file uploaded", image_url=None)

    file = request.files['file']

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict
        img_array = preprocess_image(filepath)
        preds = model.predict(img_array)
        pred_label = LABELS[np.argmax(preds)]

        image_url = f"/{filepath}"
        return render_template("result.html", result=pred_label, image_url=image_url)
    else:
        return render_template("result.html", result="Invalid file type", image_url=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
