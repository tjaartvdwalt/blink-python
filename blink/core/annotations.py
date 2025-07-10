from blink.core.frame.data import FrameData
from blink.core.blink.data import BlinkData
from blink.core.state import BlinkState
from blink.core.state import EyeState


class Annotations:
    def __init__(self, output_file: str):
        self.__output_file = output_file

        with open(self.__output_file, "w") as f:
            f.write("#start\n")

    def annotate_frame(self, data: FrameData):
        if data.full_blink_state == BlinkState.start or data.full_blink_state == BlinkState.continued:
            blink_no = data.full_blinks
        else:
            blink_no = -1

            # line =

            # [value for value in iterable if condition]

        # line = f"{{data.frame_number}}:"

        non_frontal = "X"
        left_eye_closed = "C" if data.left_eye_state == EyeState.closed else "X"
        left_eye_not_visible = 'X'

        right_eye_closed = "C" if data.right_eye_state == EyeState.closed else "X"
        right_eye_not_visible = 'X'

        face_str = ":::"
        # face = data.getFace()
        # if face:
        #     face_str = f"{face.x}:{face.y}:{face.width}:{face.height}"

        left_eye_str = "::"
        # left_eye = data.getLeftEye()
        # if left_eye:
        #     left_eye_str = f"{left_eye.x}:{left_eye.y +(left_eye.height / 2)}:{left_eye.x + left_eye.width}:{left_eye.y + left_eye.height / 2}"

        right_eye_str = "::"
        # right_eye = data.getRightEye()
        # if right_eye:
        #     right_eye_str = f"{right_eye.x}:{right_eye.y +(right_eye.height / 2)}:{right_eye.x + right_eye.width}:{right_eye.y + right_eye.height / 2}"

        line = f"{data.frame_number}:{blink_no}:{non_frontal}:{left_eye_closed}:{left_eye_not_visible}:{right_eye_closed}:{right_eye_not_visible}:{face_str}:{left_eye_str}:{right_eye_str}\n"

        print(line)
        with open(self.__output_file, "a") as f:
            f.write(line)
        


