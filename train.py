#!/usr/bin/env python
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from sid import nn
from sid import utils

path_train = os.path.join('input', 'train')
width = 128
height = 128
channels = 1
batch_size = 8
seed = 1
progress = True if "PROGRESS" in os.environ else False

print('Getting and resizing train images and masks...')
x, y, _ = utils.get_data(path_train, width, height, channels, True,
                         progress=progress)

model = nn.model(width, height, channels)

datagen_args = dict(
    rotation_range=10,
    width_shift_range=0.5,
    height_shift_range=0.5,
    zoom_range=0.5,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1,
)
datagen_x = ImageDataGenerator(**datagen_args)
datagen_y = ImageDataGenerator(**datagen_args)
gen_train = zip(
    datagen_x.flow(x, batch_size=batch_size, subset="training", seed=seed),
    datagen_y.flow(y, batch_size=batch_size, subset="training", seed=seed),
)
gen_validation = zip(
    datagen_x.flow(x, batch_size=batch_size, subset="validation", seed=seed),
    datagen_y.flow(y, batch_size=batch_size, subset="validation", seed=seed),
)

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
model.fit_generator(gen_train,
                    epochs=60,
                    steps_per_epoch=(len(x) * 0.9),
                    validation_data=gen_validation,
                    validation_steps=(len(x) * 0.1),
                    callbacks=[earlystopper, checkpointer])
