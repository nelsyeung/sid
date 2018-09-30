from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.models import Model, load_model
from keras.optimizers import Adam
from tqdm import tqdm
import pandas as pd
import numpy as np

from sid import loss
from sid import metric
from sid import nn
from sid import utils
from sid.globals import file_model, debug

x_train, x_valid, y_train, y_valid = utils.get_train(
    preprocess=2, validation_split=0.2, seed=1)

epochs = 50
batch_size = 32

# Fit with binary crossentropy loss
model = nn.model()
model.compile(loss='binary_crossentropy', optimizer=Adam(0.01),
              metrics=[metric.mean_iou])
model_checkpoint = ModelCheckpoint(file_model, monitor='mean_iou',
                                   verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='mean_iou', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.0001)
history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    batch_size=batch_size, epochs=epochs,
                    callbacks=[model_checkpoint, reduce_lr], verbose=2)

# Fit with Lovasz loss
model = load_model(file_model,
                   custom_objects={'mean_iou': metric.mean_iou})
# Remove later activation layer
model = Model(model.layers[0].input, model.layers[-1].input)
# lovasz_loss need input range (-inf, +inf), so cancel the last "sigmoid"
# activation Then the default threshod for pixel prediction is 0 instead of
# 0.5, as in mean_iou2.
model.compile(loss=loss.lovasz_loss, optimizer=Adam(0.01),
              metrics=[metric.mean_iou2])
early_stopping = EarlyStopping(monitor='val_mean_iou2', patience=20, verbose=1,
                               mode='max')
model_checkpoint = ModelCheckpoint(file_model, monitor='val_mean_iou2',
                                   verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_mean_iou2', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.0001)
history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    batch_size=batch_size, epochs=epochs,
                    callbacks=[model_checkpoint, reduce_lr], verbose=2)

model = load_model(file_model,
                   custom_objects={'mean_iou2': metric.mean_iou2,
                                   'lovasz_loss': loss.lovasz_loss})
preds_valid = utils.predict(model, x_valid)

# Scoring for last model, choose threshold by validation data.
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was
# removed.
thresholds = np.log(thresholds_ori / (1 - thresholds_ori))
loop = tqdm(thresholds) if debug else thresholds
ious = np.array([metric.iou_metric_batch(y_valid, preds_valid > threshold)
                 for threshold in loop])
print(ious)
threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

ids, x_test, x_sizes = utils.get_test()
preds_test = utils.predict(model, x_test)

pred_dict = {idx: utils.rl_encode(
    np.round(utils.resize(preds_test[i], x_sizes[i]) > threshold_best))
    for i, idx in enumerate(ids)}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
