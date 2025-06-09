import cv2 as cv
import dlib
import math
import numpy as np
import os
from icecream import ic


class CascadeLoadException(Exception):
    pass


class FaceDetectException(Exception):
    pass


class EAR:
    def __init__(
        self,
        cascade_file=os.path.join(
            os.path.dirname(__file__),
            "haarcascade_frontalface_alt.xml",
        ),
        facemarks_file=os.path.join(
            os.path.dirname(__file__),
            "shape_predictor_68_face_landmarks.dat",
        ),
    ):
        self.cascade = self.__load_cascade(cascade_file)
        self.facemarks_model = self.__load_facemarks(facemarks_file)
        self.__frame = None
        self.__frame_prev = None
        self.__points = None
        self.__points_detected = None
        self.__points_prev = None
        self.__points_prev_detected = None
        self.__eye_distance = None

    def __load_cascade(self, file):
        face_cascade = cv.CascadeClassifier()
        if not face_cascade.load(file):
            raise CascadeLoadException(f"Could not load cascade file: {file}")

        return face_cascade

    def __load_facemarks(self, file):
        predictor = dlib.shape_predictor(file)
        # facemark = cv.face.createFacemarkLBF()
        # facemark.loadModel(file)

        return predictor

    def __face_detect(self, frame):
        frame_gray = cv.equalizeHist(frame)

        faces = self.cascade.detectMultiScale(frame_gray)

        if len(faces) > 0:
            max_idx = max(((v[2] * v[3]), i) for i, v in enumerate(faces))[1]
            face = faces[max_idx]

            return face
        else:
            raise FaceDetectException("No face detected!")

    def __dist(self, p1, p2):
        return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

    def __fit(self, frame):
        face = self.__face_detect(frame)
        drect = dlib.rectangle(face[0], face[1], face[0] + face[2], face[1] + face[3])
        landmarks = self.facemarks_model(frame, drect).parts()

        # print(face)
        # print(self.__facemarks)
        return landmarks

    def __inter_eye_distance(self, predict):
        leftEyeLeftCorner = (predict[36].x, predict[36].y)
        rightEyeRightCorner = (predict[45].x, predict[45].y)
        distance = cv.norm(np.array(rightEyeRightCorner) - np.array(leftEyeLeftCorner))
        distance = int(distance)
        ic(rightEyeRightCorner)
        ic(leftEyeLeftCorner)
        return distance

    def __stabilize_points(self, landmarks):
        # Handling the first frame of video differently,for the first frame copy the current frame points
        if self.__frame_prev is None:
            self.__frame_prev = self.__frame.copy()
            self.__points_prev = []
            self.__points_detected_prev = []
            [self.__points_prev.append((p.x, p.y)) for p in landmarks]
            [self.__points_detected_prev.append((p.x, p.y)) for p in landmarks]

        # If not the first frame, copy points from previous frame.
        else:
            self.points_prev = []
            self.__points_detected_prov = []
            self.__points_prev = self.__points
            self.__points_detected_prev = self.__points_detected

        self.__points = []
        self.__points_detected = []
        [self.__points.append((p.x, p.y)) for p in landmarks]
        [self.__points_detected.append((p.x, p.y)) for p in landmarks]

        # Convert to numpy float array
        points_arr = np.array(self.__points, np.float32)
        points_prev_arr = np.array(self.__points_prev, np.float32)

        # If eye distance is not calculated before
        # print(self.__eye_distance)
        # ic(self.__eye_distance)
        if not self.__eye_distance:
            self.__eye_distance = self.__inter_eye_distance(landmarks)

        sigma = self.__eye_distance * self.__eye_distance / 400
        ic(sigma)
        s = 2 * int(self.__eye_distance / 4) + 1

        #  Set up optical flow params
        lk_params = dict(
            winSize=(s, s),
            maxLevel=5,
            criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 20, 0.03),
        )
        # Python Bug. Calculating pyramids and then calculating optical flow results in an error. So directly images are used.
        # ret, imGrayPyr = cv.buildOpticalFlowPyramid(self.__frame, (s, s), 5)

        # print(imGrayPyr)
        points_arr, status, err = cv.calcOpticalFlowPyrLK(self.__frame_prev, self.__frame, points_prev_arr, points_arr, **lk_params)

        # Converting to float
        points_arr_float = np.array(points_arr, np.float32)

        # Converting back to list
        points = points_arr_float.tolist()

        # Final landmark points are a weighted average of
        # detected landmarks and tracked landmarks
        for k in range(0, len(landmarks)):
            d = cv.norm(
                np.array(self.__points_detected_prev[k])
                - np.array(self.__points_detected[k])
            )
            alpha = math.exp(-d * d / sigma)
            val = (1 - alpha) * np.array(
                self.__points_detected[k]
            ) + alpha * np.array(points[k])
            self.__points[k] = (int(val[0]), int(val[1]))
        return self.__points

    def calc(self, frame, stabilize=False):
        self.__frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        landmarks = self.__fit(self.__frame)
        if stabilize:
            points = self.__stabilize_points(landmarks)
            unstable_points = []
            [unstable_points.append((p.x, p.y)) for p in landmarks]

            # ic(points)
            # ic(unstable_points)
        else:
            points = []
            [points.append((p.x, p.y)) for p in landmarks]

        right_vert_1 = self.__dist(points[37], points[41])
        right_vert_2 = self.__dist(points[38], points[40])
        left_vert_1 = self.__dist(points[43], points[47])
        left_vert_2 = self.__dist(points[46], points[46])

        left_horz = self.__dist(points[36], points[39])
        right_horz = self.__dist(points[42], points[45])

        left_aspect_ratio = (left_vert_1 + left_vert_2) / (2 * left_horz)
        right_aspect_ratio = (right_vert_1 + right_vert_2) / (2 * right_horz)

        self.__frame_prev = self.__frame

        return (left_aspect_ratio, right_aspect_ratio, points)

    # def landmarks(self, frame, stabilize=False):
    #     self.__frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #
    #     landmarks = self.__fit(self.__frame)
    #
    #     if stabilize:
    #         points = self.__stabilize_points(landmarks)
    #     else:
    #         points = []
    #         [points.append((p.x, p.y)) for p in landmarks]
    #
    #     return facemark
