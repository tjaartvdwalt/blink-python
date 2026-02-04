import os
import cv2
from pathlib import Path
from blink.core.frame.window import FrameWindow
from blink.core.frame.data import FrameData

WINDOW_SIZE = 13


def calc_blink_frames(annotation_file: str):
    with open(annotation_file) as f:
        lines = f.readlines()

    entries = []
    for line in lines:
        if len(line.split(':')) < 3:
            continue

        (frame_number, blink, _) = line.split(":", 2)

        if (blink) != "-1":
            entries.append(int(frame_number))

    return entries


def calc_ear(
    input_video: str,
    annotation_file: str,
    out_dir: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_data"
    ),
):
    video = cv2.VideoCapture(input_video)

    if not video.isOpened():
        raise Exception(f"Unable to open video file {input_video}")

    blink_out_file = Path(out_dir, f"blink_{Path(input_video).stem}.ear")
    non_blink_out_file = Path(out_dir, f"non_blink_{Path(input_video).stem}.ear")

    blink_f = open(blink_out_file, "w")
    non_blink_f = open(non_blink_out_file, "w")

    window = FrameWindow(WINDOW_SIZE)

    blink_frames = calc_blink_frames(annotation_file)

    frame_number = 0
    while video.isOpened():
        ret, frame = video.read()
        frame_number += 1

        if not ret:
            break

        frame_data = FrameData(frame_number, frame)

        window.put(frame_data)

        if window.full():
            left_ear = []
            right_ear = []
            els = window.list
            for el in els:
                left_ear.append(str(el.left_EAR))
                right_ear.append(str(el.right_EAR))

        
            if "None" in left_ear or "None" in right_ear:
                continue

            if window.cur.frame_number in blink_frames:
                blink_f.write(f'{" ".join(left_ear)}\n')
                blink_f.write(f'{" ".join(right_ear)}\n')
            else:
                non_blink_f.write(f'{" ".join(left_ear)}\n')
                non_blink_f.write(f'{" ".join(right_ear)}\n')

        # if cv2.waitKey(1) == "q":
        #     break
        #
        # # else:
        # #     out.write(frame)
        #
    video.release()
    blink_f.close()
    non_blink_f.close()
