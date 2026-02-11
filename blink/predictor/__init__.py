from blink.core.frame.window import FrameWindow
# from blink.core.frame.data import FrameData
from blink.core.state import EyeState
from cv2.dnn import Net
import numpy as np
from importlib.resources import files, as_file

import cv2

def predict(window: FrameWindow):
    if not window.full():
        return

    pred_left = np.zeros([1, window.size], dtype=np.float32)
    pred_right = np.zeros([1, window.size], dtype=np.float32)

    for i, el in enumerate(window.list):
        pred_left[0][i] = el.left_EAR
        pred_right[0][i] = el.right_EAR

    net = cv2.dnn.readNet(files("blink.models") / 'blink.onnx')
    # blob_left = cv2.dnn.blobFromImage(pred_left)
    # blob_right = cv2.dnn.blobFromImage(pred_right)
    # print(pred_left)

    net.setInput(pred_left)
    outputs_l = net.forward(
        net.getUnconnectedOutLayersNames())

    mat_l = outputs_l[0][0]

    if mat_l[0] < mat_l[1]:
        window.cur.left_eye_state = EyeState.closed
    else:
        window.cur.left_eye_state = EyeState.open

    net.setInput(pred_right)
    outputs_r = net.forward(
        net.getUnconnectedOutLayersNames())

    mat_r = outputs_r[0][0]

    if mat_r[0] < mat_r[1]:
        window.cur.right_eye_state = EyeState.closed
    else:
        window.cur.right_eye_state = EyeState.open
