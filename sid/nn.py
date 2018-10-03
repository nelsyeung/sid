from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    UpSampling2D,
    add,
    concatenate,
)
from keras.applications import ResNet50
from keras.models import Model
import os

from .globals import (
    width,
    height,
    channels,
    debug,
    debug_dir,
)


def upsample_block(input_tensor, filters, kernel_size=3):
    bn_axis = 3  # Channels last format

    x = UpSampling2D()(input_tensor)
    x = Conv2D(filters[0], kernel_size, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)

    shortcut = UpSampling2D()(input_tensor)
    shortcut = Conv2D(filters[1], 2, padding='same')(shortcut)
    shortcut = BatchNormalization(axis=bn_axis)(shortcut)

    x = add([x, shortcut])
    return x


def identity_block(input_tensor, filters, kernel_size=3):
    bn_axis = 3  # Channels last format

    x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def agent_add(encoder, decoder, filters):
    x = Conv2D(filters, 1, activation='relu', padding='same')(encoder)
    x = add([x, decoder])
    x = Activation('relu')(x)
    return x


def model(start_neurons=64, dropout_ratio=0.5):
    model = ResNet50(False, input_shape=(width, height, channels))

    for layer in model.layers:
        layer.trainable = False

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2),
                              padding='same')(model.output)
    uconv4 = concatenate([deconv4, model.get_layer('activation_40').output])
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None,
                    padding='same')(uconv4)

    x = identity_block(uconv4, start_neurons * 8)
    x = identity_block(x, start_neurons * 8)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2),
                              padding='same')(x)
    uconv3 = concatenate([deconv3, model.get_layer('activation_22').output])
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None,
                    padding='same')(uconv3)
    x = identity_block(uconv3, start_neurons * 4)
    x = identity_block(x, start_neurons * 4)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2),
                              padding='same')(x)
    uconv2 = concatenate([deconv2, model.get_layer('activation_10').output])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None,
                    padding='same')(uconv2)
    x = identity_block(uconv2, start_neurons * 2)
    x = identity_block(x, start_neurons * 2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2),
                              padding='same')(x)
    uconv1 = concatenate([deconv1, model.get_layer('activation_1').output])
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None,
                    padding='same')(uconv1)
    x = identity_block(uconv1, start_neurons * 1)
    x = identity_block(x, start_neurons * 1)

    outputs = Conv2D(1, (1, 1), padding='same')(x)
    outputs = Activation('sigmoid')(outputs)

    model = Model(model.input, outputs)

    if debug:
        with open(os.path.join(debug_dir, 'model.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
