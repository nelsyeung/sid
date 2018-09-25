#!/usr/bin/env python
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
import os

from sid import nn
from sid import utils
from sid.globals import width, height, channels, file_model, seed, progress

path_train = os.path.join('input', 'train')
batch_size = 8

print('Getting and resizing train images and masks...')
x_train, x_valid, y_train, y_valid, _ = utils.get_data(
    path_train, width, height, channels, True, validation_split=0.1,
    stratify=True, seed=seed, progress=progress)

datagen_args = dict(
    rotation_range=20,
    width_shift_range=0.5,
    height_shift_range=0.5,
    zoom_range=0.5,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
gen_train, gen_valid = utils.preprocess_image(
    datagen_args, x_train, y_train, x_valid, y_valid, batch_size, seed,
)

model = nn.model(width, height, channels, load=True)
early_stopping = EarlyStopping(patience=5, verbose=1)
model_checkpoint = ModelCheckpoint(file_model, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001,
                              verbose=1)
model.fit_generator(gen_train,
                    epochs=60,
                    steps_per_epoch=len(x_train),
                    validation_data=gen_valid,
                    validation_steps=len(x_valid),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])
