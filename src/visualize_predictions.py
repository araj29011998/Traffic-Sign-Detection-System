import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd

def load_data_for_prediction(train_csv_path, train_image_dir, img_size=(32, 32), batch_size=10):
    """
    Loads the data using the paths from the CSV file for prediction.
    
    Args:
    - train_csv_path (str): Path to the CSV file.
    - train_image_dir (str): Directory path where images are stored.
    - img_size (tuple): Size to which the images will be resized.
    - batch_size (int): Number of images per batch.
    
    Returns:
    - validation_generator: The data generator for validation data.
    """
    # Load CSV data
    data = pd.read_csv(train_csv_path)
    data['ClassId'] = data['ClassId'].astype(str)

    # Data preprocessing (only rescaling for predictions)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    validation_generator = datagen.flow_from_dataframe(
        dataframe=data,
        directory=train_image_dir,
        x_col='Path',
        y_col='ClassId',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True  # Ensure shuffle is true to avoid repeated images
    )

    return validation_generator

def plot_images_with_predictions(model, generator, class_names, num_images=10):
    """
    Plots images with actual and predicted labels.
    
    Args:
    - model: Trained model for predictions.
    - generator: Data generator with images and labels.
    - class_names (list): List of class names for mapping indices to labels.
    - num_images (int): Number of images to display.
    """
    plt.figure(figsize=(15, 15))

    # Get batch of images and labels from the generator
    images, labels = next(generator)

    # Handle the case where fewer images are returned
    num_images_to_show = min(num_images, len(images))

    # Predict on the batch of images
    predictions = model.predict(images)

    for i in range(num_images_to_show):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # Increase brightness by scaling pixel values
        brightened_image = np.clip(images[i] * 1.5, 0, 1)  # Brighten the image by 50%

        # Display the brightened image
        plt.imshow(brightened_image)

        # Get true label and predicted label
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(predictions[i])

        # Display actual and predicted labels
        plt.xlabel(f"Actual: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define paths
    train_csv_path = 'C:/Users/hp/Desktop/self learning/misc/Train.csv'
    train_image_dir = 'C:/Users/hp/Desktop/self learning/misc/'

    # Load the validation data generator for predictions
    validation_generator = load_data_for_prediction(train_csv_path, train_image_dir)

    # Load the trained model
    model_path = 'C:/Users/hp/Desktop/self learning/misc/models/traffic_sign_model.h5'  # Adjust the model path
    model = load_model(model_path)

    # Class names (assuming 43 classes for traffic signs)
    class_names = [str(i) for i in range(43)]

    # Plot images with actual and predicted labels
    plot_images_with_predictions(model, validation_generator, class_names, num_images=10)
