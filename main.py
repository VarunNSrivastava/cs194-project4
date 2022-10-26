import os

import matplotlib.pyplot as plt
import part_a, part_b, bells
import scipy.interpolate as skinterp
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
import random
from harris import *
from image_tools import *
np.seterr(divide='ignore')

from time import time


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
            grey_region0 = to_gray(im0[region0[:, 0], region0[:, 1], :])

            best_corr = -1
            best_x, best_y = x1, y1
            for x in range(x1 - 10, x1 + 11):
                for y in range(y1 - 10, y1 + 11):
                    region1 = lattice((y, x))
                    grey_region1 = to_gray(im1[region1[:, 0], region1[:, 1], :])
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


def ANMS(img, N=500):
    """
    Finds strongest N interest points according to r.
    Binary searches the range of r values using
    get_harris_corners until N interest points are achieved.

    It is mathematically clear that this approach achieves
    the same results as that described in the Brown et al paper
    but with o(log) times more iterations.
    """
    gray_img = normalize(to_gray(img))
    L = 1
    R = 2 * max(img.shape)
    num_points = -1
    m = num_points
    i = 0
    while L < R and m != (L + R) // 2:
        m = (L + R) // 2
        h, coords = get_harris_corners(gray_img, r=m)
        num_points = coords.shape[1]
        # print(f"Binary search with bounds {L}, {R} yielded {num_points} points")
        if num_points > N:
            L = m
        elif num_points < N:
            R = m
        # print(f"At iteration {i} m is at {m} and n is at {num_points}")
        i += 1

    h, coords = get_harris_corners(gray_img, r=m)

    # need to make coords.shape[1] exactly N;
    while coords.shape[1] != N:
        worst_r = np.argmin(h[coords[0], coords[1]])
        coords = np.hstack((coords[:, :worst_r],  coords[:, worst_r+1:]))
    return h, coords


def extract_features(img, coords):
    """
    Given an img and a 2 x n coords vector of interest points,
    return a 64 x n description vector of the interest points.
    """
    n = coords.shape[1]
    height, width, _ = img.shape
    descriptions = np.ones((64, n))
    for i in range(n):
        x, y = coords[:, i]
        min_x, max_x = max(0, x - 20), min(height - 1, x + 20)
        min_y, max_y = max(0, y - 20), min(width - 1, y + 20)
        large_patch = to_gray(img[min_x:max_x, min_y:max_y])
        desc_vector = resize(large_patch, (8, 8)).reshape((64, ))
        normalized_desc = (desc_vector - np.mean(desc_vector)) / np.std(desc_vector)
        descriptions[:, i] = normalized_desc

    return descriptions

def match_features(features1, features2, threshold=0.7):
    """
    Takes in two k x n feature vectors and computes correspondences
    between them based on nearest neighbors in R^k and Lowe thresholding.

    Returns a tuple of lists of indices of correspondences between the
    feature vectors.
    """

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(features1.T)
    distances, indices = nbrs.kneighbors(features2.T)
    points2 = np.array(np.nonzero(distances[:, 0] / distances[:, 1] < threshold)).reshape(-1)
    points1 = indices[points2, 0].reshape(-1)

    return points1, points2

def RANSAC(keypts1, keypts2, num_samples=3000, epsilon=0.5):
    """
    Computes 4-RANSAC on a set of corresponding keypts. In particular, samples
    4 pairs of keypoints and computes the corresponding number of
    inliers from the resulting homography. The transformation corresponding to
    the maximum number of inliers is kept.
    """
    assert keypts1.shape == keypts2.shape

    n = keypts2.shape[1]
    if n < 4:
        return np.ones((6, 1))
    correspondences = np.ones((6, 4))
    most_correct = -1
    best_sack = [0, 1, 2, 3]
    for i in range(num_samples):
        indices = np.random.choice(n, 4)
        correspondences[:2, :] = keypts1[:, indices]
        correspondences[3:5, :] = keypts2[:, indices]

        H = computeH(correspondences)
        preimage = np.vstack((keypts1, np.ones(n)))

        image = np.matmul(H, preimage)

        diff = np.linalg.norm(image[0:2, :] / image[2, :] - keypts2, axis=0)
        num_correct = np.count_nonzero(diff < epsilon)
        if num_correct > most_correct:
            most_correct = num_correct
            best_sack = indices

    ### using best sack to generate final correspondence of inliers ###
    correspondences[:2, :] = keypts1[:, best_sack]
    correspondences[3:5, :] = keypts2[:, best_sack]

    H = computeH(correspondences)
    preimage = np.vstack((keypts1, np.ones((n))))

    image = np.matmul(H, preimage)
    inliers = np.array(np.nonzero(np.linalg.norm(image[0:2, :] / image[2, :] - keypts2, axis=0) < epsilon)).reshape(-1)

    return np.vstack((keypts1[:, inliers], np.ones(most_correct), keypts2[:, inliers], np.ones(most_correct)))


def find_correspondences(im1, im2, threshold=0.7, num_samples=3000, epsilon=0.5):
    """
    Automatically derives the correspondence matrix between image 1 and image 2.
    """

    ### ANMS selection of corners ###
    _, sigpts1 = ANMS(im1)
    _, sigpts2 = ANMS(im2)


    ### Feature Descriptor extraction ###
    feature_vector1 = extract_features(im1, sigpts1)
    feature_vector2 = extract_features(im2, sigpts2)

    ### Feature Matching ###
    indices1, indices2 = match_features(feature_vector1, feature_vector2, threshold)

    keypts1 = sigpts1[:, indices1]
    keypts2 = sigpts2[:, indices2]

    ### RANSAC ###
    return RANSAC(keypts1, keypts2, num_samples, epsilon)


def stitch(im0, im1, correspondence, overwrite=False):
    """
    Stitches together im0 and im1 based on correspondence matrix.
    """
    im0_warped, shift_r, shift_c = warp(im0, correspondence)

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

    return stitched, shift_r, shift_c


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
        while not self.finished():
            if self.center:
                self.center = self.center - 1
            else:
                self.images[0], self.images[1] = self.images[1], self.images[0]
                self.correspondences[0] = np.vstack((self.correspondences[0][3:, :], self.correspondences[0][:3, :]))

            stitched, shift_r, shift_c = stitch(self.images[0], self.images[1], self.correspondences[0], overwrite)

            self.images.pop(0)
            self.images[0] = stitched

            self.correspondences.pop(0)
            if len(self.correspondences) > 0:
                self.correspondences[0][0, :] += shift_r
                self.correspondences[0][1, :] += shift_c

    def find_correspondences(self):
        for i in range(self.length() - 1):
            corr_i = find_correspondences(self.images[i], self.images[i + 1])
            self.correspondences.append(corr_i)

    def select_correspondences(self, N, auto=True):
        """
        Calls a user selection for correspondences between each image.
        """
        for i in range(self.length() - 1):
            corr_i = select_correspondences(self.images[i], self.images[i + 1], N, auto)
            self.correspondences.append(corr_i)

    def length(self):
        return len(self.images)

    def finished(self):
        return self.length() == 1

    def __getitem__(self, index):
        return self.images[index]

    def show(self):
        plt.imshow(self[0])
        plt.show()

    def save(self, file):
        plt.imsave(file, self[0])


def main():
    # Part A
    ### rectification ###
    part_a.pencil_box()
    part_a.skull()
    part_a.walt_jr()

    # ### mosaics ###
    part_a.varun()
    part_a.building()
    part_a.anthropology()
    part_a.floating()

    # Part B
    ### automosaics ###
    part_b.varun()
    part_b.building()
    part_b.anthropology()

    ### Bells and Whistles ###
    bells.detect_panoramics()


if __name__ == '__main__':
    main()
