from flask import Flask, request, render_template, redirect, url_for
import os
from tf_keras import models
from PIL import Image
import numpy as np

model = models.load_model("cat_dog_classifier.h5")


def preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


def predict_image(image_path):
    input_image = preprocess_image(image_path)
    prediction = model.predict(input_image)
    print(prediction)
    if prediction[0] > 0.5:
        return "It's a dog!"
    else:
        return "It's a cat!"


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def upload_page():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']

    if file.filename == '':
        return 'No selected file'

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    if os.path.exists(filepath):
        result = predict_image(filepath)
        print(result)
        return render_template('view.html', filename=f"/static/uploads/{filename}", result=result)
    else:
        return 'File type not allowed'


if __name__ == '__main__':
    app.run(debug=True)
