# from blink.core.blink.data import BlinkData
from blink.detector.landmark import LandmarkDetector
from blink.core.state import EyeState, BlinkState


class FrameData:
    def __init__(self, frame_number: int, frame):
        self.__frame_number = frame_number
        self.__frame = frame

        self.__eye_points = LandmarkDetector.detect(frame)

        self.__left_EAR = None
        self.__right_EAR = None
        if len(self.__eye_points) >= 8:
            (left_EAR, right_EAR) = LandmarkDetector.calc_EAR(self.__eye_points)
            self.__left_EAR = left_EAR
            self.__right_EAR = right_EAR

        self.__left_eye_state = EyeState.undefined
        self.__right_eye_state = EyeState.undefined

        self.__left_blink_state = BlinkState.undefined
        self.__right_blink_state = BlinkState.undefined
        self.__full_blink_state = BlinkState.undefined

        self.__left_blinks: int = -1
        self.__right_blinks: int = -1
        self.__full_blinks: int = -1

    # face: tuple[int]
    # left_eye: tuple[int]
    # right_eye: tuple[int]
    #
    #
    # fps: float
    # frame_number: int

    @property
    def eye_points(self):
        return self.__eye_points

    @property
    def frame_number(self):
        return self.__frame_number

    @property
    def frame(self):
        return self.__frame

    @property
    def left_EAR(self):
        return self.__left_EAR

    @property
    def right_EAR(self):
        return self.__right_EAR

    @property
    def left_eye_state(self):
        return self.__left_eye_state

    @left_eye_state.setter
    def left_eye_state(self, left_eye_state):
        self.__left_eye_state = left_eye_state

    @property
    def right_eye_state(self):
        return self.__right_eye_state

    @right_eye_state.setter
    def right_eye_state(self, right_eye_state):
        self.__right_eye_state = right_eye_state

    @property
    def eyes_state(self):
        if (
            self.__left_eye_state == EyeState.closed
            and self.__right_eye_state == EyeState.closed
        ):
            return EyeState.closed
        else:
            return EyeState.open

    @property
    def left_blink_state(self):
        return self.__left_blink_state

    @left_blink_state.setter
    def left_blink_state(self, left_blink_state):
        self.__left_blink_state = left_blink_state

    @property
    def right_blink_state(self):
        return self.__right_blink_state

    @right_blink_state.setter
    def right_blink_state(self, right_blink_state):
        self.__right_blink_state = right_blink_state

    @property
    def full_blink_state(self):
        return self.__full_blink_state

    @full_blink_state.setter
    def full_blink_state(self, full_blink_state):
        self.__full_blink_state = full_blink_state

    @property
    def left_blinks(self):
        return self.__left_blinks

    @left_blinks.setter
    def left_blinks(self, blinks):
        self.__left_blinks = blinks

    @property
    def right_blinks(self):
        return self.__right_blinks

    @right_blinks.setter
    def right_blinks(self, blinks):
        self.__right_blinks = blinks

    @property
    def full_blinks(self):
        return self.__full_blinks

    @full_blinks.setter
    def full_blinks(self, blinks):
        self.__full_blinks = blinks

    def __repr__(self):
        return f"""
left eye state:  {self.__left_eye_state}
right eye state: {self.__right_eye_state}

left blink:      {self.__left_blinks}
right blink:     {self.__right_blinks}
full blink:      {self.__full_blinks}
"""
