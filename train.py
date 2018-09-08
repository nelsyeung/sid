#!/usr/bin/env python
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Input,
    Lambda,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model
from tqdm import tqdm
# import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


path_train = os.path.join('input', 'train')
path_test = os.path.join('input', 'test')
im_width = 128
im_height = 128
im_chan = 1

train_ids = next(os.walk(os.path.join(path_train, 'images')))[2]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan),
                   dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, im_chan),
                   dtype=np.bool)

print('Getting and resizing train images and masks ... ')

for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    img = imread(os.path.join(path, 'images', id))[:, :, 1]
    X_train[n] = resize(img, (128, 128, 1), mode='constant',
                        preserve_range=True, anti_aliasing=False)
    mask = imread(os.path.join(path, 'masks', id))
    Y_train[n] = resize(mask, (128, 128, 1), mode='constant',
                        preserve_range=True, anti_aliasing=False)

print('Done!')

# Build U-Net model
inputs = Input((im_height, im_width, 1))
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-sid.h5', verbose=1,
                               save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8,
                    epochs=30, callbacks=[earlystopper, checkpointer])
