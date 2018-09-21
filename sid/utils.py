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


def get_data(path, width, height, channels, target=False, progress=False):
    """Return images data and, if target is True, masks target data.

    Args:
        path (string): Folder path containing images and masks.
        width (int): Input image width.
        height (int): Input image height.
        channels (int): Input image number of channels.
        target (bool, optional): Whether to return target data (masks).
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
    # The use of x and y is to match the Keras Model method's arguments.
    x = np.zeros((len(ids), height, width, channels), dtype=np.uint8)
    if target:
        y = np.zeros((len(ids), height, width, channels), dtype=np.bool)
    loop = tqdm(enumerate(ids), total=len(ids)) if progress else enumerate(ids)

    for n, id in loop:
        image = skimage.io.imread(os.path.join(path, 'images', id))[:, :, 1]
        sizes.append([image.shape[0], image.shape[1]])
        x[n] = resize(image, (width, height, channels))

        if target:
            mask = skimage.io.imread(os.path.join(path, 'masks', id))
            y[n] = resize(mask, (width, height, channels))

    if target:
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
