#!/usr/bin/python3

import os
from pathlib import Path

import cv2 as cv
import typer

from utils.ear import EAR, FaceDetectException
from utils.eyeblink_entry import EyeBlinkEntry, MalformedEntryError


def generate_frame_window(ear, video_file, starting_frame):
    cap = cv.VideoCapture(video_file)
    left_window = []
    right_window = []
    cap.set(cv.CAP_PROP_POS_FRAMES, starting_frame)

    for _ in range(0, 13):
        _, frame = cap.read()

        try:
            (l_ear, r_ear) = ear.calc(frame)
            left_window.append(l_ear)
            right_window.append(r_ear)
        except FaceDetectException:
            frame_nr = cap.get(cv.CAP_PROP_POS_FRAMES)
            print(f"No face detected at frame: {frame_nr}")

    return (left_window, right_window)


def annotation_blink_state(annotation_file: str):
    with open(annotation_file) as f:
        lines = f.readlines()

    entries = {}
    for line in lines:
        try:
            entry = EyeBlinkEntry()
            entry.parse_line(line)
            entries[entry.frame_id] = True if entry.blink_id >= 0 else False
        except MalformedEntryError:
            pass

    return entries


def main(
    video_file: str,
    annotation_file: str,
    data_dir: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_data"
    ),
):
    ear = EAR()

    blink_dict = annotation_blink_state(annotation_file)
    blink_out_file = Path(data_dir, f"blink_{Path(video_file).stem}.ear")
    non_blink_out_file = Path(data_dir, f"non_blink_{Path(video_file).stem}.ear")
    blink_f = open(blink_out_file, "w")
    non_blink_f = open(non_blink_out_file, "w")

    print(f"processing: {video_file} using annotations from: {annotation_file}")

    l_ear_out = []
    r_ear_out = []

    window_size = 13

    cap = cv.VideoCapture(video_file)

    prev_frame = None

    if not cap.isOpened:
        print("--(!)Error opening video capture")
        exit(0)
    while True:
        _, frame = cap.read()

        if frame is None:
            print("--(!) No captured frame -- Skipping frame!")
            break

        frame_nr = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        if frame_nr % 100 == 0:
            print(f"Processing frame: {frame_nr}")

        first_frame = frame_nr - window_size
        cur_frame = frame_nr - (window_size // 2)
        last_frame = frame_nr

        try:
            (l_ear, r_ear) = ear.calc(frame)
            l_ear_out.append(l_ear)
            r_ear_out.append(r_ear)
        except FaceDetectException:
            l_ear_out.append(None)
            r_ear_out.append(None)

        if cur_frame >= (window_size):
            l_window = l_ear_out[first_frame:last_frame]
            r_window = r_ear_out[first_frame:last_frame]

            if blink_dict.get(cur_frame, False):
                for window in [l_window, r_window]:
                    if None not in window:
                        s = " ".join([str(i) for i in window])
                        blink_f.write(f"{s}\n")
            else:
                for window in [l_window, r_window]:
                    if None not in window:
                        s = " ".join([str(i) for i in window])
                        non_blink_f.write(f"{s}\n")

        prev_frame = frame

if __name__ == "__main__":
    typer.run(main)
