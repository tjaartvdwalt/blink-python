import math
import os

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()




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
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker_v2_with_blendshapes.task")
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.__detector = vision.FaceLandmarker.create_from_options(options)

        # self.cascade = self.__load_cascade(cascade_file)
        # self.facemarks_model = self.__load_facemarks(facemarks_file)
        self.__frame = None
        self.__facemarks = None

    # def __load_cascade(self, file):
    #     face_cascade = cv.CascadeClassifier()
    #     if not face_cascade.load(file):
    #         raise CascadeLoadException(f"Could not load cascade file: {file}")
    #
    #     return face_cascade
    #
    # def __load_facemarks(self, file):
    #     predictor = dlib.shape_predictor(file)
    #     # facemark = cv.face.createFacemarkLBF()
    #     # facemark.loadModel(file)
    #
    #     return predictor
    #
    # def __face_detect(self, frame):
    #     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #     frame_gray = cv.equalizeHist(frame_gray)
    #
    #     faces = self.cascade.detectMultiScale(frame_gray)
    #
    #     if len(faces) > 0:
    #         max_idx = max(((v[2] * v[3]), i) for i, v in enumerate(faces))[1]
    #         face = faces[max_idx]
    #
    #         return face
    #     else:
    #         raise FaceDetectException("No face detected!")
    #
    def __dist(self, p1, p2):
        return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

    def __plot(self, frame, facemarks):
        height = frame.shape[0]
        width = frame.shape[1]


        # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), self.__facemarks)

        ll = facemarks[33]
        lr = facemarks[133]
        lt1 = facemarks[158]
        lb1 = facemarks[153]
        lt2 = facemarks[160]
        lb2 = facemarks[144]

        # chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
        # chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]

        cv.circle(frame, (int(ll.x * width), int(ll.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(lr.x * width), int(lr.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(lt1.x * width), int(lt1.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(lb1.x * width), int(lb1.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(lt2.x * width), int(lt2.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(lb2.x * width), int(lb2.y * height)), 1, (0, 0, 255), -1)

        rl = facemarks[362]
        rr = facemarks[263]
        rt1 = facemarks[387]
        rb1 = facemarks[373]
        rt2 = facemarks[385]
        rb2 = facemarks[380]

        cv.circle(frame, (int(rl.x * width), int(rl.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(rr.x * width), int(rr.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(rt1.x * width), int(rt1.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(rb1.x * width), int(rb1.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(rt2.x * width), int(rt2.y * height)), 1, (0, 0, 255), -1)
        cv.circle(frame, (int(rb2.x * width), int(rb2.y * height)), 1, (0, 0, 255), -1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        plt.imshow(rgb_frame)
        plt.show()


    def __fit(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        self.__facemarks = self.__detector.detect(mp_image)

        if len(self.__facemarks.face_landmarks) > 0:
            return self.__facemarks.face_landmarks[0]
        else:
            raise FaceDetectException("No face detected!")


        #     face = self.__face_detect(frame)
        #     drect = dlib.rectangle(
        #         face[0], face[1], face[0] + face[2], face[1] + face[3]
        #     )
        #     self.__facemarks = self.facemarks_model(frame, drect)
        #
        # # print(face)
        # # print(self.__facemarks)
        # return self.__facemarks

    def calc(self, frame):
        facemark = self.__fit(frame)

        # self.__plot(frame, facemark)
        
        height = frame.shape[0]
        width = frame.shape[1]
        
        ll = (int(facemark[33].x * width), int(facemark[33].y * height))
        lr = (int(facemark[133].x * width), int(facemark[133].y * height))
        lt1 = (int(facemark[158].x * width), int(facemark[158].y * height))
        lb1 = (int(facemark[153].x * width), int(facemark[153].y * height))
        lt2 = (int(facemark[160].x * width), int(facemark[160].y * height))
        lb2 = (int(facemark[144].x * width), int(facemark[144].y * height))

        rl = (int(facemark[362].x * width), int(facemark[362].y * height))
        rr = (int(facemark[263].x * width), int(facemark[263].y * height))
        rt1 = (int(facemark[386].x * width), int(facemark[386].y * height))
        rb1 = (int(facemark[374].x * width), int(facemark[374].y * height))
        rt2 = (int(facemark[386].x * width), int(facemark[386].y * height))
        rb2 = (int(facemark[374].x * width), int(facemark[374].y * height))

        right_vert_1 = self.__dist(rt1, rb1)
        right_vert_2 = self.__dist(rt2, rb2)
        left_vert_1 = self.__dist(lt1, lb1)
        left_vert_2 = self.__dist(lt2, lb2)

        left_horz = self.__dist(ll, lr)
        right_horz = self.__dist(rl, rr)

        left_aspect_ratio = (left_vert_1 + left_vert_2) / (2 * left_horz)
        right_aspect_ratio = (right_vert_1 + right_vert_2) / (2 * right_horz)

        return (left_aspect_ratio, right_aspect_ratio)

    def eye_marks(self, frame):
        facemark = self.__fit(frame)

        return facemark
