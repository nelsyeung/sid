from keras.applications.resnet50 import ResNet50
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
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


def upsample_block(input_tensor, filters):
    x = Conv2D(filters[0], (3, 3), padding='same',
               kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2),
                               kernel_initializer='he_normal')(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)

    x = add([x, shortcut])
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
    model = ResNet50(include_top=False, input_shape=(width, height, channels))

    # # 128 -> 64
    # c1 = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    # c1 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid',
    #             kernel_initializer='he_normal', name='conv1')(c1)
    # c1 = BatchNormalization(axis=3, name='bn_conv1')(c1)
    # c1 = Activation('relu')(c1)

    # # 64 -> 32
    # p1 = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(c1)
    # p1 = MaxPooling2D((3, 3), strides=(2, 2))(p1)
    # l1 = downsample_block(p1, 3, [64, 64, 256], stage=2, block='a',
    #                       strides=(1, 1))
    # l1 = residual_block(l1, 3, [64, 64, 256], stage=2, block='b')
    # l1 = residual_block(l1, 3, [64, 64, 256], stage=2, block='c')

    # # 32 -> 16
    # c2 = downsample_block(l1, 3, [128, 128, 512], stage=3, block='a')
    # c2 = residual_block(c2, 3, [128, 128, 512], stage=3, block='b')
    # c2 = residual_block(c2, 3, [128, 128, 512], stage=3, block='c')
    # c2 = residual_block(c2, 3, [128, 128, 512], stage=3, block='d')

    # # 16 -> 8
    # c3 = downsample_block(c2, 3, [256, 256, 1024], stage=4, block='a')
    # c3 = residual_block(c3, 3, [256, 256, 1024], stage=4, block='b')
    # c3 = residual_block(c3, 3, [256, 256, 1024], stage=4, block='c')
    # c3 = residual_block(c3, 3, [256, 256, 1024], stage=4, block='d')
    # c3 = residual_block(c3, 3, [256, 256, 1024], stage=4, block='e')
    # c3 = residual_block(c3, 3, [256, 256, 1024], stage=4, block='f')

    # # 8 -> 4
    # cm = downsample_block(c3, 3, [512, 512, 2048], stage=5, block='a')
    # cm = residual_block(cm, 3, [512, 512, 2048], stage=5, block='b')
    # cm = residual_block(cm, 3, [512, 512, 2048], stage=5, block='c')

    # 4 -> 8
    um = upsample_block(model.output, [512, 256])
    um = concatenate([um, model.get_layer('activation_40').output])
    um = Conv2D(256, (3, 3), padding='same')(um)

    # 8 -> 16
    u3 = residual_block(um, 3, [256, 256, 256], stage=6, block='f')
    u3 = residual_block(u3, 3, [256, 256, 256], stage=6, block='e')
    u3 = residual_block(u3, 3, [256, 256, 256], stage=6, block='d')
    u3 = residual_block(u3, 3, [256, 256, 256], stage=6, block='c')
    u3 = residual_block(u3, 3, [256, 256, 256], stage=6, block='b')
    u3 = upsample_block(u3, [256, 128])
    u3 = concatenate([u3, model.get_layer('activation_22').output])
    u3 = Conv2D(128, (3, 3), padding='same')(u3)

    # 16 -> 32
    u2 = residual_block(u3, 3, [128, 128, 128], stage=7, block='d')
    u2 = residual_block(u2, 3, [128, 128, 128], stage=7, block='c')
    u2 = residual_block(u2, 3, [128, 128, 128], stage=7, block='b')
    u2 = upsample_block(u2, [128, 64])
    u2 = concatenate([u2, model.get_layer('activation_10').output])
    u2 = Conv2D(64, (3, 3), padding='same')(u2)

    # 32 -> 64
    u1 = residual_block(u2, 3, [64, 64, 64], stage=8, block='c')
    u1 = residual_block(u1, 3, [64, 64, 64], stage=8, block='b')
    u1 = upsample_block(u1, [64, 64])
    u1 = concatenate([u1, model.get_layer('activation_1').output])
    u1 = Conv2D(64, (3, 3), padding='same')(u1)

    # 64 -> 128
    x = residual_block(u1, 3, [64, 64, 64], stage=9, block='c')
    x = residual_block(x, 3, [64, 64, 64], stage=9, block='b')
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(model.input, x)

    if debug:
        with open(os.path.join(debug_dir, 'model.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
