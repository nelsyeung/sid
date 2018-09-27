from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import os
import skimage.io
import skimage.transform


def resize(image, output_shape):
    """Standard resize function so that all images are resized the same way.

    Args:
        image (ndarray): Input image.
        output_shape (tuple or ndarray): Size of the generated output image.

    Returns:
        ndarray: Resized image.
    """
    return skimage.transform.resize(image, output_shape,
                                    mode='constant', preserve_range=True,
                                    anti_aliasing=False)


def mask_class(mask):
    """Return the class of a mask's coverage for stratification."""
    for i in range(11):
        if (np.sum(mask) / float(12.8 * 128)) <= i:
            return i


def get_data(path, width, height, channels, target=False, validation_split=0.0,
             stratify=False, seed=None, progress=False):
    """Return images data and, if target is True, masks target data.

    Args:
        path (string): Folder path containing images and masks.
        width (int): Input image width.
        height (int): Input image height.
        channels (int): Input image number of channels.
        target (bool, optional): Whether to return target data (masks).
            Defaults to False.
        validation_split (float, optional): Fraction of images reserved for
            validation (strictly between 0 and 1).
        stratify (bool, optional): Whether to stratify data by target coverage.
            Defaults to False.
        seed (int, optional): Provide random_state to train_test_split.
            Defaults to False.
        progress (bool, optional): Whether to log the progress.
            Defaults to False.

    Returns:
        ndarray: Numpy array of data (images).
        ndarray: Numpy array of target data (masks). Returns only if `target`
            is True
        list(int): List of width and height of data (images).
    """
    ids = os.listdir(os.path.join(path, 'images'))
    sizes = []
    classes = [] if stratify else None
    # The use of x and y is to match the Keras Model method's arguments.
    x = np.zeros((len(ids), height, width, channels), dtype=np.uint8)
    if target:
        y = np.zeros((len(ids), height, width, 1), dtype=np.bool)
    loop = tqdm(enumerate(ids), total=len(ids)) if progress else enumerate(ids)

    for n, id in loop:
        image = skimage.io.imread(os.path.join(path, 'images', id))
        sizes.append([image.shape[0], image.shape[1]])
        x[n] = resize(image, (width, height, channels))

        if target:
            mask = skimage.io.imread(os.path.join(path, 'masks', id))
            y[n] = resize(mask, (width, height, 1))

            if stratify:
                classes.append(mask_class(y[n]))

    if target:
        if validation_split:
            x_train, x_valid, y_train, y_valid = train_test_split(
                x, y, test_size=validation_split, random_state=seed,
                stratify=classes)

            return x_train, x_valid, y_train, y_valid, sizes

        return x, y, sizes

    return x, sizes


def rlenc(image):
    """Return the run-length encoded string of an image."""
    runs = []  # List of run lengths
    r = 0  # The current run length
    pos = 1  # Count starts from 1 per WK

    for c in image.reshape(image.shape[0] * image.shape[1], order='F'):
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # If last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    z = ''

    for run in runs:
        z += '{} {} '.format(run[0], run[1])

    return z[:-1]


def zip(*iterables):
    """Python 3 zip function for Python 2 as it needs to return an iterator."""
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]

    while iterators:
        result = []

        for it in iterators:
            elem = next(it, sentinel)

            if elem is sentinel:
                return

            result.append(elem)

        yield tuple(result)


def preprocess_image(datagen_args, x_train, y_train, x_valid=None,
                     y_valid=None, batch_size=8, seed=1):
    """Preprocess image using Keras Image Preproessing.

    Args:
        datagen_args (dict): Dictionary containing arguments for Keras
            ImageDataGenerator
        x_train (ndarray): Numpy array of data (images).
        y_train (ndarray): Numpy array of target data (masks).
        x_valid (ndarray, optional): Numpy array of validation data (images).
            Defaults to None.
        y_valid (ndarray, optional): Numpy array of validationtarget data
            (masks). Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 8.
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        iterator: Keras flow iterator of preprocessed training data.
        iterator: Keras flow iterator of preprocessed validation data. Returns
            only if x_valid and y_valid is set.
    """
    datagen_x = ImageDataGenerator(**datagen_args)
    datagen_y = ImageDataGenerator(**datagen_args)
    gen_train = zip(
        datagen_x.flow(x_train, batch_size=batch_size, seed=seed),
        datagen_y.flow(y_train, batch_size=batch_size, seed=seed),
    )

    if x_valid is not None and y_valid is not None:
        datagen_x = ImageDataGenerator(**datagen_args)
        datagen_y = ImageDataGenerator(**datagen_args)
        gen_valid = zip(
            datagen_x.flow(x_valid, batch_size=batch_size, seed=seed),
            datagen_y.flow(y_valid, batch_size=batch_size, seed=seed),
        )

        return gen_train, gen_valid

    return gen_train
