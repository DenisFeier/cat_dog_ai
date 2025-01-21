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


test_image_path = "cat.10.jpg"
result = predict_image(test_image_path)
print(f"Prediction for the image: {result}")
