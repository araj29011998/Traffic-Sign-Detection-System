import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_csv_path, train_image_dir, img_size=(32, 32), batch_size=32):
    """
    Loads the training and validation data using the paths from the CSV file.
    
    Args:
    - train_csv_path (str): Path to the training CSV file.
    - train_image_dir (str): Directory path where training images are stored.
    - img_size (tuple): Size to which the images will be resized.
    - batch_size (int): Number of images per batch for the data generator.
    
    Returns:
    - train_generator: The data generator for training data.
    - validation_generator: The data generator for validation data.
    """
    
    # Step 1: Load the CSV data
    train_data = pd.read_csv(train_csv_path)

    # Convert ClassId to string type
    train_data['ClassId'] = train_data['ClassId'].astype(str)

    # Data augmentation and preprocessing for training and validation sets
    datagen = ImageDataGenerator(
        rescale=1./255,   # Normalize the pixel values between 0 and 1
        validation_split=0.2)  # 20% of the data for validation

    # Load the training data from the CSV using the paths in 'Path' column
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=train_image_dir,  # Path to the folder containing the 'Train' folder
        x_col='Path',   # Column in CSV with image paths
        y_col='ClassId',  # Column in CSV with the label
        target_size=img_size,  # Resize the images to 32x32 (standard size for traffic signs)
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')  # Specify that this is the training subset

    # Load the validation data from the CSV using the paths in 'Path' column
    validation_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=train_image_dir,
        x_col='Path',  
        y_col='ClassId',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')  # Specify that this is the validation subset

    return train_generator, validation_generator
