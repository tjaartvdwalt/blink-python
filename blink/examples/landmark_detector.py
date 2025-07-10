import cv2
from blink.core.frame.data import FrameData
from blink.detector.landmark import LandmarkDetector
from blink.core.frame.window import FrameWindow
from blink.utils import plot

def main(target=0, end="!"):

    MAX_SIZE = 13

    detector = LandmarkDetector()

    window = FrameWindow(MAX_SIZE)

    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frame_number = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_data = FrameData(frame_number, frame)
        frame_number += 1

        window.put(frame_data)

        ann_frame = plot.plot_points(frame, frame_data.eye_points)
        cv2.imshow("feed", ann_frame)
        #
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
