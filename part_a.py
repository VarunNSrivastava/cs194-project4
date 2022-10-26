from main import *


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
    m.show()
    m.save("out/building.jpeg")


def anthropology():
    anthro1 = plt.imread("lib/anthro1.jpeg")
    anthro2 = plt.imread("lib/anthro2.jpeg")
    anthro3 = plt.imread("lib/anthro3.jpeg")
    m = Mosaic([anthro1, anthro2, anthro3])
    m.select_correspondences(N=8)
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
