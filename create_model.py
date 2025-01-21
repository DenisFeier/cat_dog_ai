from tf_keras import layers, models, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_with_debug(image, label):
    try:
        # Ensure valid image preprocessing
        image = tf.image.resize(image, [128, 128])
        return image, label
    except Exception as e:
        tf.print("Error processing image:", e)
        return None, None  # Skip this image


train_dir = "data/train"
val_dir = "data/validation"

train_dataset = utils.image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

train_dataset = train_dataset.map(preprocess_with_debug)

val_dataset = utils.image_dataset_from_directory(
    val_dir,
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

val_dataset = val_dataset.map(preprocess_with_debug)

normalization_layer = layers.Rescaling(1. / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)


loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


model.save("cat_dog_classifier.h5")
print("Model saved as 'cat_dog_classifier.h5'")


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

