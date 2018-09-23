#!/usr/bin/env python
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os

from sid import metric
from sid import nn
from sid import utils

path_train = os.path.join('input', 'train')
file_model = 'model.h5'
width = 128
height = 128
channels = 1
batch_size = 8
seed = os.environ['SEED'] if 'SEED' in os.environ else 1
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
datagen_x = ImageDataGenerator(**datagen_args)
datagen_y = ImageDataGenerator(**datagen_args)
gen_train = utils.zip(
    datagen_x.flow(x_train, batch_size=batch_size, seed=seed),
    datagen_y.flow(y_train, batch_size=batch_size, seed=seed),
)
gen_valid = utils.zip(
    datagen_x.flow(x_valid, batch_size=batch_size, seed=seed),
    datagen_y.flow(y_valid, batch_size=batch_size, seed=seed),
)

if os.path.exists(file_model):
    print('Model file exists, loading ' + file_model)
    model = load_model('model.h5',
                       custom_objects={'mean_iou': metric.mean_iou})
else:
    model = nn.model(width, height, channels)

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
