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

def combined_subsample_dataset(x_train, y_train, x_test, y_test, training_percentage=0.7, reduction_percentage=1.0):
    """
    Combine the training and testing datasets, reduce the size according to the specified reduction percentage,
    shuffle, and then split according to the specified training percentage.

    Parameters:
    x_train (np.array): Training data features.
    y_train (np.array): Training data labels.
    x_test (np.array): Testing data features.
    y_test (np.array): Testing data labels.
    training_percentage (float): The percentage of the combined data to use as the new training set.
    reduction_percentage (float): The percentage of the combined data to retain.

    Returns:
    np.array: New training data features.
    np.array: New training data labels.
    np.array: New testing data features.
    np.array: New testing data labels.
    """
    # Combine the datasets
    x_combined = np.concatenate((x_train, x_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0)

    # Reduce the size of the combined dataset
    reduction_indices = np.random.choice(np.arange(x_combined.shape[0]), size=int(reduction_percentage * x_combined.shape[0]), replace=False)
    x_combined = x_combined[reduction_indices]
    y_combined = y_combined[reduction_indices]

    # Shuffle the reduced dataset
    combined_indices = np.arange(x_combined.shape[0])
    np.random.shuffle(combined_indices)
    x_combined = x_combined[combined_indices]
    y_combined = y_combined[combined_indices]

    # Compute split indices based on the specified training percentage
    num_total_samples = x_combined.shape[0]
    num_training_samples = int(training_percentage * num_total_samples)

    # Split the datasets into new training and testing sets
    x_train_new = x_combined[:num_training_samples]
    y_train_new = y_combined[:num_training_samples]
    x_test_new = x_combined[num_training_samples:]
    y_test_new = y_combined[num_training_samples:]

    return x_train_new, y_train_new, x_test_new, y_test_new


def normalize_img(image, label):
  return ((tf.cast(image, tf.float32) / 255.0), label)

def resize(image, label):
  return (tf.image.resize(image, (Config.IMAGE_SIZE.value, 227)), label)
