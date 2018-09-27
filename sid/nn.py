from keras.applications import ResNet50
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    UpSampling2D,
    add,
)
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
import os

from sid import metric


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
    x = Activation('relu')(x)

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


def model(width, height, channels, file_model='model.h5', load=False,
          summary=False):
    """Return a U-Net neural network model using Keras.

    Args:
        width (int): Input image width.
        height (int): Input image height.
        channels (int): Input image number of channels.
        file_model (string, optional): Model file name. Defaults to model.h5.
        load (bool, optional): Load from previous model.h5 if exists.
            Defaults to False
        gpus (int, optional): Number of GPUs. Defaults to 0.
        summary (bool, optional): Whether to print a summary model.
            Defaults to False.

    Returns:
        Model: The Keras neural network model.
    """
    if load and os.path.exists(file_model):
        print('Model file exists, loading ' + file_model)
        model = load_model('model.h5',
                           custom_objects={'mean_iou': metric.mean_iou})
    else:
        model = ResNet50(False, input_shape=(width, width, channels))

        for layer in model.layers:
            layer.trainable = False

        x = Conv2D(512, 1, activation='relu', padding='same')(model.output)

        x = identity_block(x, 512)
        x = identity_block(x, 512)
        x = identity_block(x, 512)
        x = identity_block(x, 512)
        x = identity_block(x, 512)
        x = upsample_block(x, [512, 256])
        x = agent_add(model.get_layer('activation_40').output, x, 256)

        x = identity_block(x, 256)
        x = identity_block(x, 256)
        x = identity_block(x, 256)
        x = upsample_block(x, [256, 128])
        x = agent_add(model.get_layer('activation_22').output, x, 128)

        x = identity_block(x, 128)
        x = identity_block(x, 128)
        x = upsample_block(x, [128, 64])
        x = agent_add(model.get_layer('activation_10').output, x, 64)

        x = identity_block(x, 64)
        x = identity_block(x, 64)
        x = upsample_block(x, [64, 64])
        x = agent_add(model.get_layer('activation_1').output, x, 64)

        x = identity_block(x, 64)
        x = identity_block(x, 64)
        x = identity_block(x, 64)

        x = Conv2DTranspose(1, 1, strides=2, padding='same',
                            activation='sigmoid')(x)

        model = Model(inputs=model.input, outputs=x)

    gpus = int(os.environ['GPUS']) if 'GPUS' in os.environ else 0

    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)

    if summary:
        model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[metric.mean_iou])

    return model
