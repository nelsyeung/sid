from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    UpSampling2D,
    add,
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


def model(start_neurons=16, dropout_ratio=0.5):
    model = ResNet50(False, input_shape=(width, height, channels))

    for layer in model.layers:
        layer.trainable = False

    x = Conv2D(512, 1, activation='relu', padding='same')(model.output)

    x = identity_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)
    x = upsample_block(x, [512, 256])
    x = agent_add(model.get_layer('activation_40').input, x, 256)

    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = upsample_block(x, [256, 128])
    x = agent_add(model.get_layer('activation_22').input, x, 128)

    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = upsample_block(x, [128, 64])
    x = agent_add(model.get_layer('activation_10').input, x, 64)

    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = upsample_block(x, [64, 64])
    x = agent_add(model.get_layer('activation_1').input, x, 64)

    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    x = Conv2DTranspose(1, 1, strides=2, padding='same')(x)
    outputs = Activation('sigmoid')(x)

    model = Model(model.input, outputs)

    if debug:
        with open(os.path.join(debug_dir, 'model.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
