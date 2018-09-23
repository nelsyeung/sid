#!/usr/bin/env python
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
import os

from sid import nn
from sid import utils

path_train = os.path.join('input', 'train')
file_model = 'model.h5'
width = 128
height = 128
channels = 1
batch_size = 8
seed = int(os.environ['SEED']) if 'SEED' in os.environ else 1
gpus = int(os.environ['GPUS']) if 'GPUS' in os.environ else 0
progress = True if 'PROGRESS' in os.environ else False

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

model = nn.model(width, height, channels, gpus=gpus, load=True)
early_stopping = EarlyStopping(patience=5, verbose=1)
model_checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001,
                              verbose=1)
model.fit_generator(gen_train,
                    epochs=60,
                    steps_per_epoch=(len(x_train)),
                    validation_data=gen_valid,
                    validation_steps=(len(x_valid)),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])
