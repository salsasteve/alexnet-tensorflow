import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from helper import resize, normalize_img , combined_subsample_dataset
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from config import Config
from alex_net import AlexNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


num_classes = 10  # Number of classes in CIFAR-10

# Get the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define the label names for the CIFAR-10 dataset
label_names = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}


# subsample the dataset
x_train, y_train, x_test, y_test = combined_subsample_dataset(x_train, y_train, x_test, y_test, training_percentage=0.7)

# Print the shape of the training and testing sets
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)


# Concatenate the training and testing sets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))

train_dataset = train_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

# Uncomment to visualize the data
# visually_check_data(x_train, y_train, train_dataset, label_names)
# train_dataset = train_dataset.shuffle(SHUFFLE_VAL)
train_dataset = train_dataset.batch(Config.BATCH_SIZE.value)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(Config.BATCH_SIZE.value)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

for (img, label) in train_dataset:
  print(img.numpy().shape, label.numpy())
  break

print("CIFAR-10 dataset has been split into 70% training and 30% testing sets.")

# DATA PREPROCESSING DONE --------------------------------------------------------------------------------------------


model = AlexNet()

model.compile(loss=CategoricalCrossentropy(),
              optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

es = EarlyStopping(patience=5,
                   monitor='loss')

model.fit(train_dataset, epochs=100, validation_data=test_dataset,
          callbacks=[es])

model.save('my_model.keras')
