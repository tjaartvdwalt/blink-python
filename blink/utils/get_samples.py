#!/usr/bin/env python

import cv2 as cv
import typer

from utils.ear import EAR, FaceDetectException


def generate_frame_window(ear, video_file, starting_frame):
    cap = cv.VideoCapture(video_file)
    left_window = []
    right_window = []
    cap.set(cv.CAP_PROP_POS_FRAMES, starting_frame)

    for _ in range(0, 13):
        _, frame = cap.read()

        try:
            l_ear, r_ear, _ = ear.calc(frame, stabilize=True)
            left_window.append(l_ear)
            right_window.append(r_ear)
        except FaceDetectException:
            frame_nr = cap.get(cv.CAP_PROP_POS_FRAMES)
            print(f"No face detected at frame: {frame_nr}")

    return (left_window, right_window)


def main(video_file: str, blink_file: str, non_blink_file: str):
    ear = EAR()

    blink_out_file = f"{blink_file}.out"
    non_blink_out_file = f"{non_blink_file}.out"

    print(f"processing: {blink_file}")
    blink_frames = []
    non_blink_frames = []

    with open(blink_file) as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith("#"):
                blink_frames.append(int(line))

    with open(non_blink_file) as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith("#"):
                non_blink_frames.append(int(line))

    next_blink_idx = 0
    next_non_blink_idx = 0

    blink_frame_windows = []
    non_blink_frame_windows = []

    cap = cv.VideoCapture(video_file)

    if not cap.isOpened:
        print("--(!)Error opening video capture")
        exit(0)
    while True:
        ret, frame = cap.read()

        if frame is None:
            print("--(!) No captured frame -- Skipping frame!")
            break

        frame_nr = cap.get(cv.CAP_PROP_POS_FRAMES)
        # print(frame_nr)

        # face = detect_face(cascade)
        # facemarks = facemarks_model.fit(frame, np.array([face]))
        # # print(facemarks)
        # facemark = facemarks[1][0][0]
        # (l_ear, r_ear) = calc_ear(facemark)
        #
        # print(f"{int(frame_nr)}: {l_ear}")

        if (
            next_non_blink_idx < len(non_blink_frames)
            and int(frame_nr) == non_blink_frames[next_non_blink_idx] - 7
        ):
            (left_window, right_window) = generate_frame_window(ear, video_file, frame_nr)

            next_non_blink_idx += 1

            non_blink_frame_windows.append(left_window)
            non_blink_frame_windows.append(right_window)

        if (
            next_blink_idx < len(blink_frames)
            and int(frame_nr) == blink_frames[next_blink_idx] - 7
        ):
            (left_window, right_window) = generate_frame_window(ear, video_file, frame_nr)

            next_blink_idx += 1

            # print(left_window)
            # print(right_window)
            if len(left_window) == 13:
                blink_frame_windows.append(left_window)
            else:
                print(f"Left EAR for frame {frame_nr} is incorrect length... ignoring")

            if len(right_window) == 13:
                blink_frame_windows.append(right_window)
            else:
                print(f"Right EAR for frame {frame_nr} is incorrect length... ignoring")

    with open(blink_out_file, "w") as f:
        for window in blink_frame_windows:
            s = " ".join([str(i) for i in window])
            f.write(f"{s}\n")

    with open(non_blink_out_file, "w") as f:
        for window in non_blink_frame_windows:
            s = " ".join([str(i) for i in window])
            f.write(f"{s}\n")


if __name__ == "__main__":
    typer.run(main)
