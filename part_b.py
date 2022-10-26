import main
import importlib
importlib.reload(main)
from main import *

def varun():
    varun_left = plt.imread("lib/varun_left.jpeg")
    varun_right = plt.imread("lib/varun_right.jpeg")
    ### all harris corners ###
    # h, coords = get_harris_corners(normalize(to_gray(varun_left)))
    # plt.imshow(varun_left)
    # plt.plot(coords[1, :], coords[0, :], ".r")
    # plt.show()

    ### ANMS refinement ###
    _, left_sigpts = ANMS(varun_left, N=250)
    # plt.imshow(varun_left)
    # plt.plot(coords[1, :], coords[0, :], ".r")
    # plt.show()

    ### Feature Descriptor extraction ###
    feature_vector_left = extract_features(varun_left, left_sigpts)

    ### Feature Matching ###
    _, right_sigpts = ANMS(varun_right)
    feature_vector_right = extract_features(varun_right, right_sigpts)
    left_indices, right_indices = match_features(feature_vector_left, feature_vector_right)

    left_keypts = left_sigpts[:, left_indices]
    right_keypts = right_sigpts[:, right_indices]

    ### RANSAC ###
    correspondence = RANSAC(left_keypts, right_keypts)

    ### Display / save output ###
    m = Mosaic([varun_left, varun_right])
    m.correspondences.append(correspondence)
    m.stitch()
    m.show()
    m.save("out/auto_varun.jpeg")

def building():
    building_top = plt.imread("lib/building_top.jpeg")
    building_middle = plt.imread("lib/building_middle.jpeg")
    building_bottom = plt.imread("lib/building_bottom.jpeg")
    m = Mosaic([building_top, building_middle, building_bottom])
    m.find_correspondences()
    m.stitch()
    m.show()
    # m.save("out/auto_building.jpeg")


def anthropology():
    anthro1 = plt.imread("lib/anthro1.jpeg")
    anthro2 = plt.imread("lib/anthro2.jpeg")
    anthro3 = plt.imread("lib/anthro3.jpeg")
    m = Mosaic([anthro1, anthro2, anthro3])
    m.find_correspondences()
    m.stitch()
    m.show()
    # m.save("out/auto_anthropology.jpeg")

