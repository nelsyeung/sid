#!/usr/bin/env python
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import os

from sid import nn
from sid import utils

path_train = os.path.join('input', 'train')
width = 128
height = 128
channels = 1
progress = True if "PROGRESS" in os.environ else False

print('Getting and resizing train images and masks...')
x, y, _ = utils.get_data(path_train, width, height, channels, True,
                         progress=progress)

model = nn.model(width, height, channels)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model.h5', verbose=1,
                               save_best_only=True)
results = model.fit(x, y, validation_split=0.1, batch_size=8,
                    epochs=30, callbacks=[earlystopper, checkpointer])