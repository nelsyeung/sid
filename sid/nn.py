from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Input,
    Lambda,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
import os

from sid import metric


def model(width, height, channels, gpus=0, load=False, summary=False):
    """Return a U-Net neural network model using Keras.

    Args:
        width (int): Input image width.
        height (int): Input image height.
        channels (int): Input image number of channels.
        gpus (int, optional): Number of GPUs. Defaults to 0.
        load (bool, optional): Load from previous model.h5 if exists.
            Defaults to False
        summary (bool, optional): Whether to print a summary model.
            Defaults to False.

    Returns:
        Model: The Keras neural network model.
    """
    file_model = 'model.h5'

    if load and os.path.exists(file_model):
        print('Model file exists, loading ' + file_model)
        model = load_model('model.h5',
                           custom_objects={'mean_iou': metric.mean_iou})
    else:
        inputs = Input((height, width, channels))
        s = Lambda(lambda x: x / 255)(inputs)

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])

    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)

    if summary:
        model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[metric.mean_iou])

    return model
