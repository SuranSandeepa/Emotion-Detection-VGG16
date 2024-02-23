# -*- coding: utf-8 -*-
"""VGG16_EmotionalRecognition.ipynb

# **Import necessary libraries and mount Google Drive**
"""

import sys
import os
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Input, concatenate
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.image import img_to_array, array_to_img

from google.colab import drive
drive.mount('/content/drive')

"""# **Read the dataset and preprocess it**"""

# Read the dataset from a CSV file
df = pd.read_csv('/content/drive/My Drive/Dataset/fer2013.csv')

X, y = [], []

# Loop through the dataset and extract pixel values and labels
for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        pixels = np.array(val, 'float32')
        emotion = row['emotion']
        X.append(pixels)
        y.append(emotion)
    except:
        print(f"error occurred at index: {index} and row: {row}")

# Convert data to NumPy arrays and perform one-hot encoding on labels
X = np.array(X, 'float32')
y = to_categorical(y, num_classes=7)

# Normalize the data
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# Reshape the data for the Convolutional Neural Network
X = X.reshape(X.shape[0], 48, 48, 1)

# Resize the images to match the input shape of VGG16
X_resized = np.array([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in X])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=42)

"""# **Define and compile the model**"""

# Define batch_size and epochs
batch_size = 32
epochs = 10

# Load VGG-16 base model with pre-trained weights
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in VGG-16
for layer in vgg_base.layers:
    layer.trainable = False

# Modify the input shape to match the grayscale images
vgg_input = Input(shape=(224, 224, 3))
x = concatenate([vgg_input, vgg_input, vgg_input])  # Convert grayscale to RGB
vgg_output = vgg_base(x)

# Add Global Average Pooling layer to reduce spatial dimensions
x = GlobalAveragePooling2D()(vgg_output)

# Add fully connected layers
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

# Output layer for emotions
predictions = Dense(7, activation='softmax')(x)

# Create a new model by combining VGG-16 base and additional layers
model = Model(inputs=vgg_input, outputs=predictions)

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

"""# **Train the model**"""

# Data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Training the model with augmented data
history = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
    shuffle=True
)

"""# Evaluate the model and save **it**"""

# Print model summary
model.summary()

# Plot training & validation accuracy values
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Saving the model to use it later on
model.save('model.h5')
files.download('model.h5')

# Saving the model architecture to a JSON file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Downloading the model architecture file
from google.colab import files
files.download('model.json')

"""# **Make predictions and evaluate the model**"""

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy percentage
accuracy_percentage = accuracy_score(y_true, y_pred_classes) * 100
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")