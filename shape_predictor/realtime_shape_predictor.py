import argparse

import dlib
import cv2

parser = argparse.ArgumentParser(description='Realtime shape predictor')
parser.add_argument('model', type=str, help='model file')
parser.add_argument('--output', '-o', type=str, help='output avi format file')
args = parser.parse_args()

predictor = dlib.shape_predictor(args.model)
detector = dlib.get_frontal_face_detector()

video_capture = cv2.VideoCapture(1)

if args.output is not None:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, 12, (640, 480))

while True:
    ret, frame = video_capture.read()

    dets = detector(frame, 1)
    for d in dets:
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)
        # Draw the face landmarks on the screen.
        for i in range(shape.num_parts):
            p = shape.part(i)
            cv2.circle(frame, (p.x, p.y), 2, (0, 0, 255), 1)
            
    cv2.imshow('Video', frame)

    if args.output is not None:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
if args.output is not None:
    out.release()
cv2.destroyAllWindows()
