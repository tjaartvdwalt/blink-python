import cv2
from blink.core.frame_data import FrameData
from blink.core.landmark_detector import LandmarkDetector
from blink.core.window import FrameWindow

def main(target, end="!"):

    MAX_SIZE = 13

    detector = LandmarkDetector()

    window = FrameWindow(MAX_SIZE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frame_number = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        ear = LandmarkDetector.calc_EAR(frame, plot=True)
        frame_data = FrameData(frame_number, frame, ear)
        frame_number += 1

        window.put(frame_data)

        detector.detect(window)

        # print(window.list)
        # annotated_image = Landmarker.draw_landmarks(mp_image.numpy_view(), landmarker_result)
        cv2.imshow("feed", frame)
        # print(landmarker_result)
        #
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
