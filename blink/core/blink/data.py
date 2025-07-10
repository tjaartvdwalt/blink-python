from dataclasses import dataclass
from enum import Enum

import mediapipe as mp
from cv2.typing import NumPyArrayFloat64, NumPyArrayNumeric
import numpy as np


class BlinkState(Enum):
    no = 0
    start = 1
    continued = 2


class EyeState(Enum):
    open = 0
    partial = 1
    closed = 2


@dataclass
class BlinkData:
    frame: NumPyArrayNumeric
    left_eye: tuple[int]
    right_eye: tuple[int]

    fps: float
    frame_number: int
    blink_state_left: BlinkState
    blink_state_right: BlinkState
    blink_state_full: BlinkState
    blinks_left: int
    blinks_right: int
    blinks: int
    eye_state_left: EyeState
    eye_state_right: EyeState
    aspect_ratio_left: float
    aspect_ratio_right: float
    
    @property
    def face(self) -> tuple[int]:
        if not self.face:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.frame)

        result = LandmarkDetector.__landmarker.detect(mp_image)
        print(result)
    #     else:
    #         return self.face
    #
    # {
    #     if (!this->face)
    #     {
    #         std::vector<cv::Rect> faces =
    #             detect(*this->frame, BlinkData::faces_cascade);
    #
    #         if (faces.size() >= 1)
    #         {
    #             // We will use the largest detected face
    #             int max_surface = 0;
    #             int max_i = 0;
    #             for (long unsigned int i = 0; i < faces.size(); i++)
    #             {
    #                 int face_surface = faces[i].width * faces[i].height;
    #                 if (face_surface > max_surface)
    #                 {
    #                     max_surface = face_surface;
    #                     max_i = i;
    #                 }
    #             }
    #
    #             cv::Rect* face = new cv::Rect(faces[max_i].x, faces[max_i].y,
    #                 faces[max_i].width, faces[max_i].height);
    #             this->face = face;
    #         }
    #     }
    #     return face;
    # }
    #
    #     return self._name
