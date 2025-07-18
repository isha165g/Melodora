import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_dir = "data/train"
test_dir = "data/test"

# Image properties
img_height, img_width = 48, 48  # Most facial datasets use 48x48
batch_size = 64

# Preprocess and load training images
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',  # FER datasets usually grayscale
    batch_size=batch_size,
    class_mode='categorical'
)

# Preprocess and load testing images
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)
