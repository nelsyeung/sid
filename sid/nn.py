from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model
import os

from .globals import (
    width,
    height,
    channels,
    debug,
    debug_dir,
)


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same',
                      activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)

    if activation:
        x = BatchActivate(x)

    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def model(start_neurons=16, dropout_ratio=0.5):
    inputs = Input((height, width, channels))

    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None,
                   padding='same')(inputs)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_ratio / 2)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None,
                   padding='same')(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_ratio)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None,
                   padding='same')(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_ratio)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None,
                   padding='same')(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_ratio)(pool4)

    convm = Conv2D(start_neurons * 16, (3, 3), activation=None,
                   padding='same')(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2),
                              padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None,
                    padding='same')(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)
    uconv4 = Dropout(dropout_ratio)(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2),
                              padding='same')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout_ratio)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None,
                    padding='same')(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2),
                              padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None,
                    padding='same')(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2),
                              padding='same')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(dropout_ratio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None,
                    padding='same')(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(uconv1)

    model = Model(inputs=[inputs], outputs=[outputs])

    if debug:
        with open(os.path.join(debug_dir, 'model.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
