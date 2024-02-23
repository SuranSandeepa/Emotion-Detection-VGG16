# Emotion-Detection-VGG16

This repository contains a deep learning model for emotion detection, built with Keras and the VGG16 architecture.

## Overview

The model is trained on the FER2013 dataset, which consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

## Model Architecture

The model uses the VGG16 architecture, pre-trained on the ImageNet dataset. The input images are converted to RGB and resized to match the input shape of VGG16. The output from VGG16 is passed through a Global Average Pooling layer to reduce spatial dimensions. This is followed by two fully connected layers with 1024 neurons each, and a final output layer with 7 neurons (one for each emotion category).

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss. Data augmentation is applied to the training data using Keras's ImageDataGenerator.

## Evaluation

The model's performance is evaluated using a confusion matrix and accuracy percentage.

## Usage

To use the model, load the saved model file (`model.h5`) and the JSON file containing the model architecture (`model.json`). You can then use the model to make predictions on your own data.
