#!/usr/bin/env python
from tensorflow.keras.models import load_model
from tqdm import tqdm, trange
import numpy as np
import os
import pandas as pd

from sid import metric
from sid import utils

path_test = os.path.join('input', 'test')
width = 128
height = 128
channels = 1

print('Getting and resizing test images...')
x, sizes_test = utils.get_data(path_test, width, height, channels)

model = load_model('model.h5',
                   custom_objects={'mean_iou': metric.mean_iou})
preds_test = model.predict(x, verbose=1)

print('Resizing predictions to original image size...')
preds_test_upsampled = []
for i in trange(len(preds_test)):
    preds_test_upsampled.append(utils.resize(
        np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1])))

print('Getting run-length encoding from predictions...')
ids = os.listdir(os.path.join(path_test, 'images'))
pred_dict = {fn[:-4]: utils.rlenc(np.round(preds_test_upsampled[i])) for i, fn
             in tqdm(enumerate(ids), total=len(ids))}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
