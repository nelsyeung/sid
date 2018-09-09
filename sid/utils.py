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
