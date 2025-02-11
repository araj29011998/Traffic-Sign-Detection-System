import os
from src.model import build_model  # Use absolute import
from src.data_loader import load_data  # Ensure data_loader is also imported using absolute path

# Define paths
train_csv_path = 'C:/Users/hp/Desktop/self learning/misc/Train.csv'
train_image_dir = 'C:/Users/hp/Desktop/self learning/misc/'

# Load data
train_generator, validation_generator = load_data(train_csv_path, train_image_dir)

# Build the model
model = build_model()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=validation_generator,
    verbose=1
)

# Save the trained model
if not os.path.exists('../models/'):
    os.makedirs('../models/')
model.save('../models/traffic_sign_model.h5')
