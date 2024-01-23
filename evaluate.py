# Evaluate the model on the validation set
loss, accuracy = model_bayes.evaluate(validation_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import  cv2

# 1. Probability Plot for a Given Image

def plot_image_probabilities(model, image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the image
    predictions = model(img_array)

    # Extract probabilities
    probs = predictions.mean().numpy()[0]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar([0, 1], probs, color=['blue', 'red'])
    plt.xticks([0, 1], ['Not Infected', 'Infected'])
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.show()
