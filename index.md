# Music Recommendation using Emotion Detection 🎵😄😢

Welcome to my AI project!  
This site documents my journey building an emotion-based music recommender.

## 🔍 Project Goals
- Detect emotions from facial images using deep learning
- Recommend music that matches the user's emotion
- Learn & apply core AI techniques in Python

## Phase 0: Environment Setup
- I've used **Anaconda** to setup a **Python Environment** for this project. Anaconda is like a toolbox for Python which has a bunch of useful libraries pre-installed. 
- A Python Environment is a self-contained workspace that has a specific set of packages/libraries installed and its own settings,isolated from other projects. By using an environment, each project gets its own clean workspace.
- It comes with a package manager-**conda**. Package manager is a tool that helps download, install, update and remove packages/libraries from the environment.   
(pip V/S conda : pip gets the latest official version of any library, whereas conda may have older versions with limited platform support, but still conda is preferred for heavy scientific libraries since we need stable and precompiled packages
NOTE: Install using conda befor pip, because pip can override some conda-installed libraries)

## Phase 1: Data Loading & Preparation
### Imports
- I've used **TensorFlow**-a popular open-source framework for building and training machine learning models.
- *ImageDataGenerator* is a utility from **Keras** that helps to load images from folders, preprocess images, create batches of images during training.
- Finally, **os** is a Python module to work with file paths and directories.

### Dataset Configuration
- I've used a grayscale emotion image dataset from **Kaggle**.
- Used TensorFlow ImageDataGenerator to normalize and preprocess images.

## Phase 2: Model Building & Training 
In this phase I've created a CNN using TensorFlow/Keras that learns to classify facial expressions from the training dataset.   
A Convolutional Neural Network (CNN) is a specialized type of deep learning algorithm primarily designed for processing and analyzing visual data, such as images and videos.
### Imports
- I've imported **Sequential** from TensorFlow. It is a type of model in Keras where layers are stacked linearly, meaning the output of one layer serves as the input to the next layer in a straightforward, ordered sequence.
- **Conv2D** Performs 2D convolution to extract features (edges, textures, etc.) from input images.
- **MaxPooling2D** Downsamples the feature maps, reducing dimensions and computation while retaining important info.
- **Flatten** Converts 2D feature maps to 1D vectors so as to feed them into Dense layers.
- **Dense** Fully connected layers that perform classification.
- **Dropout** Randomly disables some neurons during training to avoid overfitting.   
All these extract features from the images and help make predictions.

### Model Architecture
- 