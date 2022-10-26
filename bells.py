from main import *

def ball():
    ### failed attempt at stitching gifs ###

    middle_frame = plt.imread(f"lib/gianni_gif/frame_35.jpeg")
    for i in range(70, 89):
        print(f"processing frame {i}")
        left_frame = add_alpha(normalize(plt.imread(f"lib/ball_left_gif/frame{i}.jpg")))[:, 340:620, :]

        right_frame = add_alpha(normalize(plt.imread(f"lib/ball_right_gif/frame{i + 2}.jpg")))[:, 340:620, :]
        correspondence = find_correspondences(left_frame, right_frame, threshold=0.5, epsilon=0.9)
        try:
            stitched, _, _ = stitch(left_frame, right_frame, correspondence)
            plt.imsave(f"out/ball_gif/frame_{i - 70}.png", stitched)
        except MemoryError:
            # indicates a degenerate stitch; shouldnt happen
            pass


def detect_panoramics():
    unordered_images = []
    filenames = []
    for filename in os.listdir("lib/unordered"):
        if filename.startswith("."):
            continue
        image_i = plt.imread(f"lib/unordered/{filename}")
        unordered_images.append(image_i)
        filenames.append(filename)
    for i in range(len(filenames)):
        print(f"{i}: \t {filenames[i]}")
    # mosaics list containing lists of mosaics
    mosaics = []
    for i in range(len(unordered_images) - 1):
        for j in range(i + 1, len(unordered_images)):
            im0 = unordered_images[i]
            im1 = unordered_images[j]
            correspondence = find_correspondences(im0, im1, threshold=.8, epsilon=2, num_samples=3500)
            # a good correspondence
            if correspondence.shape[1] >= 6:
                print(f"Correspondence found with {i, j}")
                found = False
                for lst in mosaics:
                    if len(lst) == 1:
                        if i == lst[0]:
                            lst.append(j)
                            found = True
                        elif j == lst[0]:
                            lst.append(i)
                            found = True
                    elif len(lst) > 1:
                        if i == lst[0]:
                            lst.insert(j, 0)
                            found = True
                        elif i == lst[-1]:
                            lst.append(j)
                            found = True
                        elif j == lst[0]:
                            lst.insert(i, 0)
                            found = True
                        elif j == lst[-1]:
                            lst.append(i)
                            found = True
                if not found:
                    mosaics.append([i, j])

    for indices in mosaics:
        print(f"Mosaic found with {indices}")
        m = Mosaic([unordered_images[i] for i in indices])
        m.find_correspondences()
        try:
            m.stitch()
            m.save(f"out/unordered/mosaic{indices[0]}.jpeg")
        except MemoryError:
            # Even though there is correspondence, panoramic impossible
            print("Couldn't create a Mosaic for given images")
