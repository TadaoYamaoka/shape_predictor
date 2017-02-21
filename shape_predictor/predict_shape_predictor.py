import os
import sys
import glob

import dlib
from skimage import io

if len(sys.argv) != 3:
    print("predict_shape_predictor.py model targets")
    exit()
model = sys.argv[1]
faces_folder = sys.argv[2]

predictor = dlib.shape_predictor(model)
detector = dlib.get_frontal_face_detector()

print("Showing detections and predictions on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
