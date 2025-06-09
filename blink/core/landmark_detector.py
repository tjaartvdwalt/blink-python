from blink.landmarker.landmarker import Landmarker
from blink.core.window import FrameWindow

import mediapipe as mp
import cv2


class LandmarkDetector:
    __landmarker = Landmarker().landmarker

    def __init__(self):
        self.__svm = None

    @staticmethod
    def plot_landmarks(image, landmarks):
        for landmark in landmarks:
            cv2.circle(image, landmark, 1, (0, 255, 0), 1)

    @staticmethod
    def __dist(p1, p2):
        pass

    @staticmethod
    def calc_EAR(cur_frame, plot=False):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cur_frame)
        result = LandmarkDetector.__landmarker.detect(mp_image)

        if len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]

            if plot:
                h, w, _ = cur_frame.shape
                points = []
                eye_points = [
                    # left eye right corner
                    landmarks[463],
                    # left eye left corner
                    landmarks[263],
                    # left eye right top
                    landmarks[386],
                    # left eye right bottom
                    landmarks[374],
                    # right eye right corner
                    landmarks[130],
                    # right eye left corner
                    landmarks[133],
                    # right eye right top
                    landmarks[159],
                    # right eye right bottom
                    landmarks[145],
                ]
                for eye_point in eye_points:
                    points.append((int(eye_point.x * w), int(eye_point.y * h)))

                LandmarkDetector.plot_landmarks(cur_frame, points)
                    # landmarks[37]
                    # landmkarks[41]
                    # if (cur->points()->size() > 0)
                    # {
                    #     float right_vert_1 = (float)cv::norm(
                    #         cur->points()->at(37) - cur->points()->at(41));
                    #     float right_vert_2 = (float)cv::norm(
                    #         cur->points()->at(38) - cur->points()->at(40));
                    #     float left_vert_1 = (float)cv::norm(
                    #         cur->points()->at(43) - cur->points()->at(47));
                    #     float left_vert_2 = (float)cv::norm(
                    #         cur->points()->at(44) - cur->points()->at(46));
                    #
                    #     float left_horz = (float)cv::norm(
                    #         cur->points()->at(36) - cur->points()->at(39));
                    #     float right_horz = (float)cv::norm(
                    #         cur->points()->at(42) - cur->points()->at(45));


    @staticmethod
    def plot_EAR(window):
        pass

    def detect(self, window: FrameWindow):
        if not window.full():
            return

        for el in window.list:
            pass
