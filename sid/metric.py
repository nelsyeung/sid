import numpy as np
import tensorflow as tf


def iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []

    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        eps = 1e-10
        iou = (np.sum(intersection > 0) + eps) / (np.sum(union > 0) + eps)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []

        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def mean_iou(y_true, y_pred):
    return tf.py_func(iou_vector, [y_true, y_pred > 0.5], tf.float64)


def mean_iou2(y_true, y_pred):
    return tf.py_func(iou_vector, [y_true, y_pred > 0], tf.float64)


def iou_metric(y_true_in, y_pred_in, print_table=False):
    """src: https://www.kaggle.com/aglotero/another-iou-metric"""
    labels = y_true_in
    y_pred = y_pred_in

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(),
                           bins=([0, 0.5, 1], [0, 0.5, 1]))
    intersection = temp1[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = (np.sum(true_positives), np.sum(false_positives),
                      np.sum(false_negatives))
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []

    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)

        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []

    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)

    return np.mean(metric)
