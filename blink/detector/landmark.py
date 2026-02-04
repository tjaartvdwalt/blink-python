# from inspect import CORO_CLOSED
from blink.landmarker.landmarker import Landmarker
import mediapipe as mp
# import cv2


import math


class LandmarkDetector:
    __landmarker = Landmarker().landmarker

    def __init__(self):
        self.__svm = None

    @staticmethod
    def detect(frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = LandmarkDetector.__landmarker.detect(mp_image)

        eye_points = []
        if len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]

            h, w, _ = frame.shape
            detect_points = [
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
            for point in detect_points:
                eye_points.append((int(point.x * w), int(point.y * h)))

        return eye_points

    @staticmethod
    def __dist(p1, p2):
        return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))

    @staticmethod
    def __calc_bbox(p1, p2, p3, p4):
        xs, ys = zip(*[p1, p2, p3, p4])
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        return ((min_x, min_y), (max_x, max_y))

    @staticmethod
    def calc_EAR(eye_points):
        left_horizontal = LandmarkDetector.__dist(eye_points[0], eye_points[1])
        left_vertical = LandmarkDetector.__dist(eye_points[2], eye_points[3])
        right_horizontal = LandmarkDetector.__dist(
            eye_points[4], eye_points[5])
        right_vertical = LandmarkDetector.__dist(eye_points[6], eye_points[7])
        left_EAR = left_vertical / left_horizontal
        right_EAR = right_vertical / right_horizontal

        left_eye_bbox = LandmarkDetector.__calc_bbox(
            eye_points[0], eye_points[1], eye_points[2], eye_points[3]
        )
        right_eye_bbox = LandmarkDetector.__calc_bbox(
            eye_points[4], eye_points[5], eye_points[6], eye_points[7]
        )

        return (left_EAR, right_EAR, left_eye_bbox, right_eye_bbox)
