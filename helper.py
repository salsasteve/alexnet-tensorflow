import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from config import Config


def visually_check_data(x_data, y_data, x_data_resized, label_names):
    # Choose an index to display an image, for example, index 0
    index = 0

    # Original image and label
    original_image = x_data[index]
    # Convert one-hot encoded label back to a single integer
    original_label = np.argmax(y_data[index])

    # Ensure the image is in the correct format for visualization
    if original_image.dtype != 'uint8':
        original_image = original_image.astype('uint8')

    # Extract the resized (and normalized) image and label
    for resized_image, label in x_data_resized.take(1):
        # Convert one-hot encoded label back to a single integer
        resized_label = np.argmax(label.numpy())  # Assuming the label is also part of the dataset
        break  # Extracted the resized image and label from the dataset

    # If the resized image is a TensorFlow tensor, convert it to a numpy array
    if hasattr(resized_image, 'numpy'):
        resized_image = resized_image.numpy()

    # After normalization, the pixel values are in [0, 1], so for visualization, we scale them back to [0, 255]
    resized_image = (resized_image * 255).astype('uint8')  # Ensure correct data type for visualization

    # Set up a figure with two subplots
    plt.figure(figsize=(8, 4))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f'Original Image: {label_names[original_label]}')  # Use the integer label to access the label name
    plt.axis('off')

    # Plot resized image
    plt.subplot(1, 2, 2)
    plt.imshow(resized_image)
    plt.title(f'Resized Image: {label_names[resized_label]}')  # Use the integer label to access the label name
    plt.axis('off')

    # Show the plot
    plt.show()

def subsample_dataset(x_train, y_train, x_test, y_test, num_training_samples=1000, num_testing_samples=200):

    # Shuffling the dataset for unbiased random sampling
    training_indices = np.arange(x_train.shape[0])
    np.random.shuffle(training_indices)  # This shuffles the array in-place

    # Select a random subset of the training data
    train_indices = training_indices[:num_training_samples]
    x_train_subset = x_train[train_indices]
    y_train_subset = y_train[train_indices]

    # Shuffle the indices of the testing data
    testing_indices = np.arange(x_test.shape[0])
    np.random.shuffle(testing_indices)  # This shuffles the array in-place

    # Select a random subset of the testing data
    test_indices = testing_indices[:num_testing_samples]
    x_test_subset = x_test[test_indices]
    y_test_subset = y_test[test_indices]

    return x_train_subset, y_train_subset, x_test_subset, y_test_subset


def normalize_img(image, label):
  return ((tf.cast(image, tf.float32) / 255.0), label)

def resize(image, label):
  return (tf.image.resize(image, (Config.IMAGE_SIZE.value, 227)), label)
