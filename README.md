# cs194-project4
Varun Neal Srivastava

3036400739

varun.neal@berkeley.edu

https://inst.eecs.berkeley.edu/~cs194-26/fa22/upload/files/proj4A/cs194-26-agh/

Project structure:
```
  - main.py: the entire python program.
    It can be run as-is, so long as the "lib/" 
    directory is populated and the "out/" directory exists. 
    Relevant classes and methods:
        + Mosaic class: stores images sequentially. Relevant methods:
            -> select_correspondences(): manually select the correspondences
                                         between adjascent images.
            -> warp(): warps the leftmost image to match the perspective
                       of the second-leftmost image, e.g. according to
                       the first correspondence matrix.
            -> stitch(): stitches the leftmost and second-leftmost image
                         together, and updates relevant correspondence
                         matrices. The plane of the leftmost image warps to
                         match the second-leftmost image.
        + computeH(): given correspondences between im0 and im1, return the 
                      homography matrix between them.
        + interp2(): naive (nearest-neighbor) interpolation method
        + part_a(): all images generated using part_a(). I recommend 
                    commenting out to select which parts of this method you would
                    like to run.
  lib/: contains some of the original images (just the jpegs). 
  out/: this is where output images go. Make sure to initialize this so that your code will run
  - README.md: this file
  - index.html: website
```

