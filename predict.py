#!/usr/bin/env python
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tqdm import tqdm, trange
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

path_test = os.path.join('input', 'test')
test_ids = next(os.walk(os.path.join(path_test, 'images')))[2]

im_width = 128
im_height = 128
im_chan = 1


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

print('Getting and resizing train images ... ')

for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    img = imread(os.path.join(path, 'images', id))[:, :, 1]
    X_train[n] = resize(img, (128, 128, 1), mode='constant',
                        preserve_range=True, anti_aliasing=False)
    mask = imread(os.path.join(path, 'masks', id))

print('Done!')
# Get and resize test images
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan),
                  dtype=np.uint8)
sizes_test = []

print('Getting and resizing test images ... ')

for n, id in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = path_test
    img = imread(os.path.join(path, 'images', id))[:, :, 1]
    sizes_test.append([img.shape[0], img.shape[1]])
    X_test[n] = resize(img, (128, 128, 1), mode='constant',
                       preserve_range=True, anti_aliasing=False)

print('Done!')

# Predict on train, val and test
model = load_model('model-sid.h5',
                   custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in trange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c) order is down-then-right, i.e.
    Fortran format determines if the order needs to be preformatted (according
    to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


pred_dict = {fn[:-4]: RLenc(np.round(preds_test_upsampled[i])) for i, fn in
             tqdm(enumerate(test_ids))}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
