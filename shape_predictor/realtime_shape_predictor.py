import os
import sys
import glob

import dlib
import cv2

if len(sys.argv) != 2:
    print("realtime_shape_predictor.py model")
    exit()
model = sys.argv[1]

predictor = dlib.shape_predictor(model)
detector = dlib.get_frontal_face_detector()

video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()

    dets = detector(frame, 1)
    for d in dets:
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)
        # Draw the face landmarks on the screen.
        for i in range(shape.num_parts):
            p = shape.part(i)
            cv2.circle(frame, (p.x, p.y), 3, (0, 0, 255), 1)
            
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
