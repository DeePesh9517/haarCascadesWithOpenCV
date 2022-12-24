# import libray
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascades', type=str, default="cascades", help='path to input directory containing haar cascades')
args = vars(ap.parse_args())

print(args)

# initialize a dictionary that maps the name of haar cascades to the filenames
datector_paths = {
    "face": "haarcascade_frontalface_default.xml",
    "eyes": "haarcascade_eye.xml",
    "smile": "haarcascade_smile.xml"
}

# initialize a dictionary to store our detectorw
print("loading haar cascades..")
detectors = {}

for (name, path) in datector_paths.items():
    # load the haar cascades from the disk and store them in detectors dict.
    path = os.path.join(args['cascades'], path)
    detectors[name] = cv2.CascadeClassifier(path)

print("starting video..")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = detectors["face"].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in face_rects:
        face_roi = gray[fY:fY + fH, fX:fX + fW]

        eye_rects = detectors["eyes"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

        smile_rects = detectors["smile"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

        for (eX, eY, eW, eH) in eye_rects:
            ptA = (fX+eX, fY+eY)
            ptB = (fX+eX+eW, fY+eY+eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)

        for (sX, sY, sW, sH) in smile_rects:
            ptA = (fX+sX, fY+sY)
            ptB = (fX+sX+sW, fY+sY+sH)
            cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)
        
        cv2.rectangle(frame, (fX, fY), (fX+fW, fY+fH), (0, 255, 0), 2)

        cv2.imshow("Frams", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cv2.destroyAllWindows()
vs.stop()
