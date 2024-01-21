import os
import random
import shutil

# Set the path to the original dataset folder
original_dataset_path = 'C:/Users/utsav/Desktop/Auto-PCOS Challenge/PCOSGen-train/images'

# Set the path to the folder where we want to save the training and testing datasets
output_path = 'C:/Users/utsav/Desktop/Auto-PCOS Challenge/PCOSGen-train-test-split/'

# Create training and testing folders
train_folder = os.path.join(output_path, 'train')
test_folder = os.path.join(output_path, 'test')

os.makedirs(train_folder, exist_ok = True)
os.makedirs(test_folder, exist_ok = True)

# Set the ratio for splitting (80:20)
split_ratio = 0.8

# Get the list of all image files in the original dataset folder
image_files = [f for f in os.listdir(original_dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

# Shuffle the list of image files
random.shuffle(image_files)

# Calculate the number of images for training and testing
num_train = int(len(image_files) * split_ratio)
num_test = len(image_files) - num_train

# Copy training images to the train folder
for i in range(num_train):
    src_path = os.path.join(original_dataset_path, image_files[i])
    dst_path = os.path.join(train_folder, image_files[i])
    shutil.copy(src_path, dst_path)

# Copy testing images to the test folder
for i in range(num_train, num_train + num_test):
    src_path = os.path.join(original_dataset_path, image_files[i])
    dst_path = os.path.join(test_folder, image_files[i])
    shutil.copy(src_path, dst_path)

print(f"Dataset split and saved successfully. Training images: {num_train}, Testing images: {num_test}")