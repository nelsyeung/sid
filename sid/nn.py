from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    add,
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


def upsample_block(input_tensor, filters, kernel_size=(3, 3)):
    x = Conv2DTranspose(filters[1], kernel_size, strides=(2, 2))(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x


def downsample_block(input_tensor, kernel_size, filters, stage, block,
                     strides=(2, 2)):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters[0], (1, 1), strides=strides,
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size, padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters[2], (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=3, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def residual_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters[0], (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def model():
    # inputs = Input((height, width, channels))
    model = InceptionResNetV2(False, input_shape=(width, height, channels))
    f = 1536

    # 2 -> 5
    um = upsample_block(model.output, [f, f/2])
    um = concatenate([um, model.get_layer('activation_162').output])
    um = Conv2D(f/2, (3, 3), padding='same')(um)
    um = Conv2D(f/2, (3, 3), padding='same')(um)

    # 5 -> 11
    u3 = upsample_block(um, [f/2, f/4])
    u3 = concatenate([u3, model.get_layer('activation_75').output])
    u3 = Conv2D(f/4, (3, 3), padding='same')(u3)
    u3 = Conv2D(f/4, (3, 3), padding='same')(u3)

    # 11 -> 24
    u2 = upsample_block(u3, [f/4, f/8], (4, 4))
    u2 = concatenate([u2, model.get_layer('activation_5').output])
    u2 = Conv2D(f/8, (3, 3), padding='same')(u2)
    u2 = Conv2D(f/8, (3, 3), padding='same')(u2)

    # 24 -> 50
    u1 = upsample_block(u2, [f/8, f/16], (4, 4))
    u1 = concatenate([u1, model.get_layer('activation_3').output])
    u1 = Conv2D(f/16, (3, 3), padding='same')(u1)
    u1 = Conv2D(f/16, (3, 3), padding='same')(u1)

    # 50 -> 101
    x = upsample_block(u1, [f/16, f/32], (3, 3))
    x = Conv2D(f/32, (3, 3), padding='same')(x)
    x = Conv2D(f/32, (3, 3), padding='same')(x)
    x = Conv2D(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(model.input, x)

    if debug:
        with open(os.path.join(debug_dir, 'model.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
