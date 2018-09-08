#!/usr/bin/env python
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tqdm import tqdm
import numpy as np
import os

from sid import nn

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

model = nn.model(im_height, im_width, im_chan)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-sid.h5', verbose=1,
                               save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8,
                    epochs=30, callbacks=[earlystopper, checkpointer])
