from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import PIL
import matplotlib.pyplot as plt
import numpy as np
import os

from .globals import (
    width,
    height,
    channels,
    progress,
    debug,
    debug_dir,
)

if not os.path.isdir(debug_dir):
    os.makedirs(debug_dir)


def resize(image, size):
    return np.array(Image.fromarray(image, 'F').resize((size[0], size[1])))


def get_train(validation_split=0.1, seed=None):
    print('Getting and resizing train images and masks...')
    path = os.path.join('input', 'train')
    path_images = os.path.join(path, 'images')
    path_masks = os.path.join(path, 'masks')
    ids = os.listdir(path_images)
    images = np.zeros((len(ids), height, width, channels), dtype=np.float32)
    masks = np.zeros((len(ids), height, width, channels), dtype=np.uint8)
    coverages = []
    classes = []
    n = 0

    for id in tqdm(ids, total=len(ids)) if progress else ids:
        image = Image.open(os.path.join(path_images, id)).convert('L')
        mask = Image.open(os.path.join(path_masks, id)).convert('L')
        image = np.array(image.resize((width, height)),
                         dtype=np.float32)[..., np.newaxis] / 255
        mask = np.array(mask.resize((width, height)),
                        dtype=np.uint8)[..., np.newaxis] / 255
        images[n] = image
        masks[n] = mask
        n += 1

        # Get coverge class of mask.
        coverages.append(10 * np.sum(mask) /
                         float(width * height * channels))

        for i in range(11):
            if coverages[-1] <= i:
                classes.append(i)
                break

    x_train, x_valid, y_train, y_valid = train_test_split(
        images, masks, test_size=validation_split, random_state=seed,
        stratify=classes)

    if debug:
        # Plot histogram of coverages and classes.
        plt.figure(figsize=(8, 6), dpi=300)
        plt.subplot(2, 2, 1)
        plt.hist(coverages, 10)
        plt.xlabel('Coverage')

        plt.subplot(2, 2, 2)
        plt.hist(classes, 10)
        plt.xlabel('Coverage class')

        plt.subplot(2, 2, 3)
        plt.scatter(coverages, classes)
        plt.xlabel('Coverage')
        plt.ylabel('Coverage class')

        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, 'salt-coverage.png'))
        plt.close()

        # Show some example input images with mask overlay.
        num_images = 60
        grid_width = 15
        grid_height = int(num_images / grid_width)
        out_file = 'input-images.png'

        print('Writing {:d} input images to {}'.format(
            num_images, out_file))
        fig, axs = plt.subplots(grid_height, grid_width,
                                figsize=(grid_width, grid_height))

        for i in range(num_images):
            image = images[i][:, :, 0]
            mask = masks[i][:, :, 0]
            coverage = coverages[i]
            ax = axs[int(i / grid_width), i % grid_width]
            ax.imshow(image, cmap='Greys')
            ax.imshow(mask, alpha=0.2, cmap='Greens')
            ax.text(width - 1, 1, round(coverage, 2), color='black',
                    ha='right', va='top')
            ax.text(1, 1, classes[i], color='black', ha='left', va='top')
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        plt.suptitle('Green: salt. Top-left: coverage class, top-right: salt '
                     'coverage')
        plt.savefig(os.path.join(debug_dir, out_file))
        plt.close()

        out_file = 'x_train-images.png'
        print('Writing {:d} x_train images to {}'.format(
            num_images, out_file))
        fig, axs = plt.subplots(grid_height, grid_width,
                                figsize=(grid_width, grid_height))

        for i in range(num_images):
            image = x_train[i][:, :, 0]
            mask = y_train[i][:, :, 0]
            ax = axs[int(i / grid_width), i % grid_width]
            ax.imshow(image, cmap='Greys')
            ax.imshow(mask, alpha=0.2, cmap='Greens')
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        plt.suptitle('Green: salt')
        plt.savefig(os.path.join(debug_dir, out_file))
        plt.close()

        out_file = 'x_valid-images.png'
        print('Writing {:d} x_valid images to {}'.format(
            num_images, out_file))
        fig, axs = plt.subplots(grid_height, grid_width,
                                figsize=(grid_width, grid_height))

        for i in range(num_images):
            image = x_valid[i][:, :, 0]
            mask = y_valid[i][:, :, 0]
            ax = axs[int(i / grid_width), i % grid_width]
            ax.imshow(image, cmap='Greys')
            ax.imshow(mask, alpha=0.2, cmap='Greens')
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        plt.suptitle('Green: salt')
        plt.savefig(os.path.join(debug_dir, out_file))
        plt.close()

    return x_train, x_valid, y_train, y_valid


def preprocess_image(image, mask, preprocess):
    images = np.zeros((preprocess, height, width, channels), dtype=np.float32)
    masks = np.zeros((preprocess, height, width, channels), dtype=np.bool)

    for i in range(preprocess):
        pimage = image
        pmask = mask

        if i > 0:
            flipv = np.random.randint(2)
            fliph = np.random.randint(2)

            if flipv:
                # Flip vertically
                pimage = pimage.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                pmask = pmask.transpose(PIL.Image.FLIP_TOP_BOTTOM)

            if fliph:
                # Flip horizontally
                pimage = pimage.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                pmask = pmask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            # Zoom
            zoom = np.random.uniform(0.8, 1.3)
            w = int(width * zoom)
            h = int(height * zoom)
            r = width - ((w/2 + width/2) - (w/2 - width/2))
            crop = (w/2 - width/2, h/2 - height/2,
                    w/2 + width/2 + r, h/2 + height/2 + r)
            pimage = pimage.resize((w, h)).crop(crop)
            pmask = pmask.resize((w, h)).crop(crop)

            # Rotate
            angle = np.random.randint(-180, 180)
            pimage = pimage.rotate(angle)
            pmask = pmask.rotate(angle)

        masks[i] = np.array(pmask)[..., np.newaxis]

        if channels == 1:
            images[i] = np.array(pimage.convert('L'))[..., np.newaxis]
        else:
            images[i] = np.array(pimage)

    return zip(images / 255, masks)


def preprocess_train(x, y, preprocess, seed=None):
    print('Preprocessing images and masks...')
    images = np.zeros((preprocess * len(x), height, width, channels),
                      dtype=np.float32)
    masks = np.zeros((preprocess * len(x), height, width, channels),
                     dtype=np.uint8)
    n = 0
    np.random.seed(seed)

    for i in trange(len(x)) if progress else range(len(x)):
        image = Image.fromarray((x[i] * 255)[:, :, 0], 'F')
        mask = Image.fromarray((y[i] * 255)[:, :, 0])

        for image, mask in preprocess_image(image, mask, preprocess):
            images[n] = image
            masks[n] = mask
            n += 1

    if debug:
        num_images = 60
        grid_width = 15
        grid_height = int(num_images / grid_width)
        out_file = 'preprocessed-images.png'
        print('Writing {:d} x_train images to {}'.format(
            num_images, out_file))
        fig, axs = plt.subplots(grid_height, grid_width,
                                figsize=(grid_width, grid_height))

        for i in range(num_images):
            image = images[i][:, :, 0]
            mask = masks[i][:, :, 0]
            ax = axs[int(i / grid_width), i % grid_width]
            ax.imshow(image, cmap='Greys')
            ax.imshow(mask, alpha=0.2, cmap='Greens')
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        plt.suptitle('Green: salt')
        plt.savefig(os.path.join(debug_dir, out_file))
        plt.close()

    return images, masks


def get_test():
    print('Getting and resizing test images...')
    path = os.path.join('input', 'test')
    path_images = os.path.join(path, 'images')
    ids = os.listdir(path_images)
    images = np.zeros((len(ids), height, width, channels), dtype=np.float32)
    sizes = np.zeros((len(ids), 2), dtype=np.uint8)
    n = 0

    for id in tqdm(ids, total=len(ids)) if debug else ids:
        image = Image.open(os.path.join(path_images, id)).convert('L')
        sizes[n][0], sizes[n][1] = image.size
        image = image.resize((width, height))
        images[n] = np.array(image)[..., np.newaxis] / 255.0
        n += 1

    return ids, images, sizes


def predict(model, x):
    x_reflect = np.array([np.fliplr(v) for v in x])
    preds = model.predict(x).reshape(-1, width, height)
    preds_refect = model.predict(x_reflect).reshape(-1, width, height)
    preds += np.array([np.fliplr(v) for v in preds_refect])
    return preds / 2


def rl_encode(image):
    pixels = image.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
