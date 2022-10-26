import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.color import rgba2rgb, rgb2gray


def to_gray(im):
    if len(im.shape) == 2:
        return im
    elif im.shape[2] == 3:
        return rgb2gray(im)
    elif im.shape[2] == 4:
        return rgb2gray(rgba2rgb(im))


def NCC(v1, v2):
    """
    returns NCC of 1d arrays v1 and v2.
    """
    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)
    return np.dot(nv1, nv2)


def lattice(guess):
    """
    returns all points in a +/-5 window around guess
    """

    return np.array([
        [int(h + guess[0]), int(w + guess[1])]
        for h in range(-5, 6)
        for w in range(-5, 6)
    ])


def gkern(kernlen, std=None):
    """
    Returns a 2D Gaussian kernel array.
    """
    if not std:
        std = 0.3 * ((kernlen - 1) * 0.5 - 1) + 0.8
    gkern1d = cv2.getGaussianKernel(kernlen, std)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def blur(img, amount, std=None):
    """
    Applies a gaussian kernal to img with size amount and std.
    """
    G = gkern(amount, std)
    if len(img.shape) == 2:
        return convolve2d(img, G, mode='same')
    r = convolve2d(img[:, :, 0], G, mode='same')
    g = convolve2d(img[:, :, 1], G, mode='same')
    b = convolve2d(img[:, :, 2], G, mode='same')

    return normalize(np.dstack([r, g, b]), hard=True)


def normalize(img):
    """
    normalizes img to [0, 1]
    """
    if np.max(img) == np.min(img):
        return img
    if len(img.shape) == 2:
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[2]):
        layer = img[:, :, i]
        if np.max(layer) == np.min(layer):
            out[:, :, i] = layer
        else:
            out[:, :, i] = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))
    return out


def add_alpha(img):
    """
    adds an alpha channel to rgb img
    """
    if img.shape[2] == 3:
        alpha = np.ones((img.shape[0], img.shape[1]))
        return np.dstack((img, alpha))
    return img


def points(im):
    """
    returns all the indices in a given image as
    points in the format [r c 1].T
    with the corners as the first four entries
    """
    corners = np.array([
        [im.shape[0] - 1, 0, 0, im.shape[0] - 1],
        [0, 0, im.shape[1] - 1, im.shape[1] - 1],
        [1, 1, 1, 1]
    ])
    inner_points = np.array([[c, h, 1] for h in np.arange(im.shape[1]) for c in np.arange(im.shape[0])]).T

    return np.hstack((corners, inner_points))
