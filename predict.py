#!/usr/bin/env python
from tqdm import tqdm, trange
import numpy as np
import os
import pandas as pd

from sid import nn
from sid import utils

path_test = os.path.join('input', 'test')
width = 128
height = 128
channels = 1
gpus = int(os.environ['GPUS']) if 'GPUS' in os.environ else 0
progress = True if "PROGRESS" in os.environ else False

print('Getting and resizing test images...')
x, sizes_test = utils.get_data(path_test, width, height, channels,
                               progress=progress)

model = nn.model(width, height, channels, load=True, gpus=gpus)
preds_test = model.predict(x, verbose=1)

print('Resizing predictions to original image size...')
preds_test_upsampled = []
preds_range = trange(len(preds_test)) if progress else range(len(preds_test))
for i in preds_range:
    preds_test_upsampled.append(utils.resize(
        np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1])))

print('Getting run-length encoding from predictions...')
ids = os.listdir(os.path.join(path_test, 'images'))
ids_enum = tqdm(enumerate(ids), total=len(ids)) if progress else enumerate(ids)
pred_dict = {fn[:-4]: utils.rlenc(np.round(preds_test_upsampled[i]))
             for i, fn in ids_enum}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
