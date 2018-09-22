#!/usr/bin/env python
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os

from sid import metric
from sid import nn
from sid import utils


def mask_class(mask):
    """Return the class of a mask's coverage for stratification."""
    for i in range(11):
        if (np.sum(mask) / float(12.8 * 128)) <= i:
            return i


path_train = os.path.join('input', 'train')
file_model = 'model.h5'
width = 128
height = 128
channels = 1
batch_size = 8
seed = os.environ['SEED'] if 'SEED' in os.environ else 1
progress = True if 'PROGRESS' in os.environ else False

print('Getting and resizing train images and masks...')
x, y, _ = utils.get_data(path_train, width, height, channels, True,
                         progress=progress)

x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=0.1, random_state=1,
    stratify=[mask_class(mask) for mask in y],
)

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
                    steps_per_epoch=(len(x) * 0.9),
                    validation_data=gen_valid,
                    validation_steps=(len(x) * 0.1),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])
