import matplotlib
from scipy.spatial import Delaunay
from skimage.draw import polygon
from skimage.color import rgba2rgb, rgb2gray
from scipy.ndimage import distance_transform_edt
import scipy.interpolate as skinterp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import cv2
from time import time
import os


def computeH(correspondences):
    """
    solves for h where A * h = b is our homography system of equations
    """
    num_points = correspondences.shape[1]
    A = np.zeros((2 * num_points, 8))
    b = np.zeros((2 * num_points, 1))
    for i in range(num_points):
        p0 = correspondences[:3, i]  # [x, y, 1]
        p1 = correspondences[3:6, i]  # [x', y', 1]
        A[2 * i, 0:3] = p0
        A[2 * i, 6:8] = [-p0[0] * p1[0], - p0[1] * p1[0]]
        A[1 + 2 * i, 3:6] = p0
        A[1 + 2 * i, 6:8] = [-p0[0] * p1[1], - p0[1] * p1[1]]

        b[2 * i] = p1[0]
        b[1 + 2 * i] = p1[1]

    h = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.vstack((h, [1])).reshape(3, 3)

    return H


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
        layer = img[:,:, i]
        if np.max(layer) == np.min(layer):
            out[:,:,i] = layer
        else:
            out[:,:,i] = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))
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
    points in the format [x y 1].T
    with the corners as the first four entries
    """
    corners = np.array([
        [im.shape[0] - 1, 0, 0, im.shape[0] - 1],
        [0, 0, im.shape[1] - 1, im.shape[1] - 1],
        [1, 1, 1, 1]
    ])
    inner_points = np.array([[c, h, 1] for h in np.arange(im.shape[1]) for c in np.arange(im.shape[0])]).T

    return np.hstack((corners, inner_points))


def interp2(warped_r, warped_c, output_quad_r, output_quad_c, im0, im0_points):
    """
    Naive, nearest-neighbor, interpolation function, going channel-by-channel.
    """
    rgb = im0[im0_points[0, :], im0_points[1, :]]

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    intp_r = skinterp.griddata((warped_r, warped_c), r, (output_quad_r, output_quad_c), method="nearest")
    intp_g = skinterp.griddata((warped_r, warped_c), g, (output_quad_r, output_quad_c), method="nearest")
    intp_b = skinterp.griddata((warped_r, warped_c), b, (output_quad_r, output_quad_c), method="nearest")
    intp_a = np.ones_like(intp_r)
    return np.dstack((intp_r, intp_g, intp_b, intp_a))


def warp(img, correspondence):
    """
    Returns the warp of the image img to match the shape of
    the given correspondence matrix correspondence.
    """
    img_points = points(img)

    corr_matrix = computeH(correspondence)
    img_warped_points = np.matmul(corr_matrix, img_points)

    warped_r = np.int0(img_warped_points[0, :] / img_warped_points[2, :])
    warped_c = np.int0(img_warped_points[1, :] / img_warped_points[2, :])

    shift_r = - min(np.min(warped_r), 0)
    shift_c = - min(np.min(warped_c), 0)

    warped_r += shift_r
    warped_c += shift_c

    warped_height = int(np.max(warped_r)) + 1
    warped_width = int(np.max(warped_c)) + 1

    # the shape of the warped image
    output_quad_r, output_quad_c = polygon(warped_r[:4], warped_c[:4])

    img_warped = np.zeros((warped_height, warped_width, 4), dtype=np.float32)

    img_warped[output_quad_r, output_quad_c] = interp2(warped_r, warped_c, output_quad_r, output_quad_c, img,
                                                       img_points)
    return img_warped, shift_r, shift_c

def select_correspondences(im0, im1, N=4, auto=True):
    """
    Tool to manually select N correspondence points between im0 and im1.
    The "auto" boolean is a naive snapping tool to adjust a correspondence point
    based on NCC score.
    """

    plt.ion()
    fig, axarr = plt.subplots(1, 2, figsize=(18, 6))
    axarr[0].imshow(im0)
    axarr[1].imshow(im1)

    correspondence = np.ones((6, N))

    for j in range(N):
        ### Select first point ###
        x0, y0 = fig.ginput(1)[0]
        x0, y0 = int(x0), int(y0)
        correspondence[0, j], correspondence[1, j] = y0, x0
        axarr[0].plot(x0, y0, '.r')
        axarr[0].annotate(j + 1, (x0, y0), xytext=(x0 + 25, y0 + 25))

        ### Select second point ###
        x1, y1 = fig.ginput(1)[0]
        x1, y1 = int(x1), int(y1)

        if auto:
            ### Automatically "snap" x, y to an adjusted value ###
            region0 = lattice((y0, x0))
            grey_region0 = rgb2gray(rgba2rgb(im0[region0[:, 0], region0[:, 1], :]))

            best_corr = -1
            best_x, best_y = x1, y1
            for x in range(x1 - 10, x1 + 11):
                for y in range(y1 - 10, y1 + 11):
                    region1 = lattice((y, x))
                    grey_region1 = rgb2gray(rgba2rgb(im1[region1[:, 0], region1[:, 1], :]))
                    score = NCC(grey_region0.flatten(), grey_region1.flatten())
                    # print(f"{(x, y)} \t score: {score}")
                    if score > best_corr:
                        best_x, best_y = x, y
                        best_corr = score
            if score > 0.99:
                x1, y1 = best_x, best_y

        correspondence[3, j], correspondence[4, j] = y1, x1
        axarr[1].plot(x1, y1, '.r')
        axarr[1].annotate(j + 1, (x1, y1), xytext=(x1 + 25, y1 + 25))

    plt.ioff()
    plt.close()

    return correspondence


class Mosaic:
    """
    Holds a collection of images.

    Supports a series of methods to stitch
    the images into a mosaic.
    """

    def __init__(self, images):
        """
        :param images: Arranged from left to right.
        Every image must have an alpha channel.
        """
        assert len(images) >= 2
        self.center = len(images) // 2
        self.images = [add_alpha(normalize(img)) for img in images]
        self.correspondences = []

    def stitch(self, overwrite=False):
        """
        Warps together the leftmost image in self.images (self.images[0])
        and the second leftmost image in self.images (self.images[1]).

        The new image will have the perspective of self.images[0] if and only if
        self.center == 0 (e.g. it will have the perspective self.images[1] otherwise).

        Removes the old correspondence matrix and updates the next
        correspondence matrix with shifted points.

        """
        assert self.length() >= 2
        if self.center:
            self.center = self.center - 1
        else:
            self.images[0], self.images[1] = self.images[1], self.images[0]
            self.correspondences[0] = np.vstack((self.correspondences[0][3:, :], self.correspondences[0][:3, :]))

        im0_warped, shift_r, shift_c = warp(self.images[0], self.correspondences[0])
        im1 = self.images[1]

        im0_height, im0_width, _ = im0_warped.shape
        im1_height, im1_width, _ = im1.shape

        stitched_height = int(max(im0_height, shift_r + im1_height))
        stitched_width = int(max(im0_width, shift_c + im1_width))

        stitched0 = np.zeros((stitched_height, stitched_width, 4))
        stitched1 = np.copy(stitched0)
        stitched0[:im0_height, :im0_width] = im0_warped
        stitched1[shift_r:shift_r + im1_height, shift_c:shift_c + im1_width] = im1

        alpha0 = stitched0[:, :, 3]
        alpha1 = stitched1[:, :, 3]
        alpha = np.logical_or(alpha0, alpha1)

        mask = np.zeros((stitched_height, stitched_width))

        if overwrite:
            # overwrites im0 on top of im1
            # alpha channel tells us where im0 is defined
            im0_alpha = im0_warped[:, :, 3]
            mask[:im0_height, :im0_width] = im0_alpha
        else:
            # cross-blends im0 and im1
            # using a distance transform as discussed in lecture
            # followed by a blur
            distance0 = distance_transform_edt(alpha0)
            distance1 = distance_transform_edt(alpha1)

            binary_mask = distance0 > distance1
            mask = blur(binary_mask, 40, std=20)

        mask = np.expand_dims(mask, axis=2)

        stitched = normalize(stitched0 * mask + stitched1 * (1 - mask))
        stitched[:, :, 3] = alpha

        self.images.pop(0)
        self.images[0] = stitched

        self.correspondences.pop(0)
        if self.length() >= 2:
            self.correspondences[0][0, :] += shift_r
            self.correspondences[0][1, :] += shift_c

    def select_correspondences(self, N, auto=True):
        for i in range(self.length() - 1):
            corr_i = select_correspondences(self.images[i], self.images[i + 1], N, auto)
            self.correspondences.append(corr_i)

    def length(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

    def show(self):
        plt.imshow(self[0])
        plt.show()

    def save(self, file):
        plt.imsave(file, self[0])


def pencil_box():
    box = normalize(plt.imread("lib/pencil_box.jpeg"))

    ### pre-defined correspondences
    correspondence = np.array([
        [160, 73, 345, 195],
        [58, 628, 103, 812],
        [1, 1, 1, 1],
        [0, 299, 0, 299],
        [0, 0, 299, 299],
        [1, 1, 1, 1]
    ])
    r, _, _ = warp(box, correspondence)
    plt.imshow(r)
    plt.show()
    # plt.imsave("out/pencil_box.jpeg", r)
    # rectified_pencil_box.stitch()

def skull():
    ambassadors_skull = normalize(plt.imread("lib/ambassadors_skull.jpeg"))

    blank = np.zeros((300, 300, 4))
    corr = select_correspondences(ambassadors_skull, blank, N=4, auto=False)
    plt.imshow(ambassadors_skull)
    plt.show()
    r, _, _ = warp(ambassadors_skull, corr)
    plt.imshow(ambassadors_skull)
    plt.show()
    print(r.shape)
    plt.imshow(r)
    plt.show()
    plt.imsave("out/skull3.jpeg", r)


def walt_jr():
    walter_white = plt.imread("lib/walter_white.jpeg")
    finger = plt.imread("lib/finger.jpeg")
    m = Mosaic([finger, walter_white])
    m.select_correspondences(N=4, auto=False)
    m.stitch(overwrite=True)
    m.show()
    m.save("out/walt_jr.jpeg")


def varun():
    varun_left = plt.imread("lib/varun_left.jpeg")
    varun_right = plt.imread("lib/varun_right.jpeg")
    m = Mosaic([varun_left, varun_right])
    m.center = 0
    m.select_correspondences(N=4)
    m.stitch()
    m.show()
    m.save("out/varun_L.jpeg")

def building():
    building_top = plt.imread("lib/building_top.jpeg")
    building_middle = plt.imread("lib/building_middle.jpeg")
    building_bottom = plt.imread("lib/building_bottom.jpeg")
    m = Mosaic([building_top, building_middle, building_bottom])
    m.select_correspondences(N=8, auto=False)
    m.stitch()
    m.stitch()
    m.show()
    m.save("out/building.jpeg")

def anthropology():
    anthro1 = plt.imread("lib/anthro1.jpeg")
    anthro2 = plt.imread("lib/anthro2.jpeg")
    anthro3 = plt.imread("lib/anthro3.jpeg")

    m = Mosaic([anthro1, anthro2, anthro3])
    m.select_correspondences(N=8)
    m.stitch()
    m.stitch()
    m.show()
    m.save("out/anthropology2.jpeg")

def floating():
    floating_top = plt.imread("lib/floating_top.jpeg")
    floating_bottom = plt.imread("lib/floating_bottom.jpeg")
    m = Mosaic([floating_bottom, floating_top])
    m.select_correspondences(N=4)
    m.stitch()
    m.show()
    m.save("out/floating2.jpeg")

def part_a():
    ### rectification ###
    # pencil_box()
    # skull()
    # walt_jr()

    ### mosaics ###
    # varun()
    # building()
    anthropology()
    # floating()


def main():
    part_a()



if __name__ == '__main__':
    main()
