from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from config import Config

# https://pub.towardsai.net/alexnet-implementation-from-scratch-667063ab5b44
# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""
Building a CNN in my own words:
    First, we build the convolution steps to automatically create the features.
    These features are generated in the convolution step in the CNN. We initially size the
    filters to some window size also called a kernal. Then we also initialize the filter values
    to some random low number and then run them over the images in a convolution pattern.
    We then run them through ReLU to remove negatives. Next, we normalize by subtracting the batch
    mean and dividing by the batch standard deviation, and then perform max pooling to reduce the
    dimensions of the filters. We repeat this process a couple of times to obtain finer-grain output
    features. The output of this is then flattened and fed into a deep neural network, where we
    perform normal neural network learning. We train the neural network after we have created the
    features and feed those in with the label to train our model.

    Super Werid:
        The features are create simultaneously with the training of the neural network. Its part of the training process.
        The features get better as the neural network learns.
"""


def AlexNet():
    # All pooling layers in the convolutional blocks use stride 2 and kernel size 3.

    input_layer = layers.Input((Config.IMAGE_SIZE.value, Config.IMAGE_SIZE.value, Config.COLOR_CHANNELS.value))

    # Block 1
    x = layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')(input_layer)
    x = layers.Lambda(lambda z: tf.nn.local_response_normalization(z, depth_radius=5, bias=2.0, alpha=0.0001, beta=0.75))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Block 2
    x = layers.ZeroPadding2D(padding=(2, 2))(x) # padding = 2 according https://pub.towardsai.net/
    x = layers.Conv2D(filters=256, kernel_size=5, strides=1, activation='relu')(x)
    # k=2=bias, n=5=depth_radius, alpha=0.0001, beta=0.75
    x = layers.Lambda(lambda z: tf.nn.local_response_normalization(z, depth_radius=5, bias=2.0, alpha=0.0001, beta=0.75))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Block 3 No normalization or pooling
    x = layers.ZeroPadding2D(padding=(1, 1))(x) # padding = 1 according https://pub.towardsai.net/
    x = layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu')(x)

    # Block 4 No normalization or pooling
    x = layers.ZeroPadding2D(padding=(1, 1))(x) # padding = 1 according https://pub.towardsai.net/
    x = layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu')(x)

    # Block 5
    x = layers.ZeroPadding2D(padding=(1, 1))(x) # padding = 1 according https://pub.towardsai.net/
    x = layers.Conv2D(256, 3, 1, activation='relu')(x)
    x = layers.Lambda(lambda z: tf.nn.local_response_normalization(z, depth_radius=5, bias=2.0, alpha=0.0001, beta=0.75))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(10, activation='softmax')(x)



    model = Model(inputs=input_layer, outputs=x)

    return model


# AlexNet().summary()
