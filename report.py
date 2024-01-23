import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_probability as tfp
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

# TensorFlow Probability layers and distributions
tfpl = tfp.layers
tfd = tfp.distributions

# Setup the dataset
data_dir = '/content/drive/MyDrive/PCOS_formatted'
img_height, img_width = 256, 256
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=False,
    vertical_flip=False,
    rotation_range=20,
    zoom_range=0.1,
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Divergence function for Bayesian layers
divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / train_generator.samples

# Build the Bayesian CNN model
model_bayes = Sequential([
    tfpl.Convolution2DReparameterization(input_shape=(255,255, 3),
                                          filters=8, kernel_size=16, activation='relu',
                                          kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                          kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                          kernel_divergence_fn=divergence_fn,
                                          bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                          bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                          bias_divergence_fn=divergence_fn),
    Conv2D(32, (3,3), activation='relu'),
     MaxPooling2D(2,2),
    Dropout(0.28),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.28),
  #  Conv2D(64, (3,3), activation='relu'),
  #  MaxPooling2D(2,2),
  #  Dropout(0.15),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.28),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.28),
    tfpl.DenseReparameterization(units=tfpl.OneHotCategorical.params_size(2), activation=None,
                                 kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                 kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                 kernel_divergence_fn=divergence_fn,
                                 bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                 bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                 bias_divergence_fn=divergence_fn
                                ),
    tfpl.OneHotCategorical(2)
])
model_bayes.summary()

# Compile the model
def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

model_bayes.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025),
                    loss=negative_log_likelihood,
                    metrics=['accuracy'])


# Train the model
epochs = 1

history = model_bayes.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1
)

# pip install reportlab

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Function to predict and gather information for a single image
def predict_and_gather_info(model, image_path):
    try:
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (255, 255))  # Resize the image to match the model's input shape
        img = img / 255.0  # Normalize the image

        # Make the image compatible with the model's input shape
        img_array = np.expand_dims(img, axis=0)

        # Predict the image
        predictions = model(img_array)

        # Extract probabilities
        probs = predictions.mean().numpy()[0]

        return img, probs

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Function to create a bar chart and save it as an image
def create_bar_chart(probs, image_filename):
    labels = ['Not Infected', 'Infected']
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close()

# Loop through the first 10 images in the training dataset
for i in range(10000, 10010):
    # Create a new PDF document for each image
    pdf_filename = f"automatic_report_{i}.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    # Add a title to the report
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 750, f"Automatic Report for Image {i}")

    image_path = f"/content/drive/MyDrive/PCOS_formatted/infected/image_{i}.jpg"  # Adjust the path accordingly

    # Predict and gather information for the image
    image, probabilities = predict_and_gather_info(model_bayes, image_path)

    # Create and save the bar chart as an image
    bar_chart_filename = f"bar_chart_{i}.png"
    create_bar_chart(probabilities, bar_chart_filename)

    # Add the image to the report
    c.drawImage(image_path, 100, 450, width=300, height=300)

    # Add the bar chart to the report
    c.drawImage(bar_chart_filename, 100, 150, width=400, height=200)

    # Add information to the report
    c.setFont("Helvetica", 12)
    c.drawString(100, 400, "Predictions:")
    c.drawString(100, 380, f"Not Infected Probability: {probabilities[0]:.4f}")
    c.drawString(100, 360, f"Infected Probability: {probabilities[1]:.4f}")

    # Save and close the PDF for this image
    c.save()

    print(f"Automatic report for Image {i} generated and saved as {pdf_filename}.")
