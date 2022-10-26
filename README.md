# cs194-project4
Varun Neal Srivastava

3036400739

varun.neal@berkeley.edu

https://inst.eecs.berkeley.edu/~cs194-26/fa22/upload/files/proj4B/cs194-26-agh/

Project structure:
```
  - main.py: the entire python program.
    It can be run as-is, so long as the "lib/" 
    directory is populated and the "out/" directory exists. 
    Relevant classes and methods:
        + Mosaic class: stores images sequentially. Relevant methods:
            -> select_correspondences(): manually select the correspondences
                                         between all adjascent images.
            -> find_correspondences(): detects the correspondences between
                                       all adjascent images.
            -> stitch(): stitches all images in the mosaic with respect to the
                         perspective of the center image
        + computeH(): given correspondences between im0 and im1, return the 
                      homography matrix between them.
        + warp(im, corr): warps the given image im according to the correspondence
                          matrix provided.
        + stitch(im0, im1, corr): stitches im0, im1 according to the give correspondence
                                  matrix. 
        + interp2(): naive (nearest-neighbor) interpolation method
        + ANMS(): implementation of ANMS using a binary search of get_harris_corners.
        + extract_features(): returns 64-dim feature vectors of a series of points on an image.
        + match_features(): matches two feature vectors using Nearest Neighbors and Lowe 
                            threshholding. 
        + RANSAC(): implementation of 4-RANSAC to robust corresponences.
        + find_correspondences(im1, im2): combines above methods to automatically find correspondences
                                          between two images.
  - part_a.py: the methods to generate Part A images
  - part_b.py: the methods to generate Part B images
  - part_b.py: the methods to generate bells and whistles images
  - image_tools.py: image tools I've made like kernals and normalizing.
  - harris.py: provided python file; get_harris corners slightly modified
  lib/: contains some of the original images (just the jpegs). 
  out/: this is where output images go. Make sure to initialize this so that your code will run
  - README.md: this file
  - index.html: website
```

