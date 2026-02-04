import cv2

from blink import predictor
from blink.core.annotations import Annotations
from blink.core.frame.data import FrameData
from blink.core.frame.window import FrameWindow
from blink.core.judge import SimpleJudge
from blink.detector.landmark import LandmarkDetector
from blink.core.state import BlinkState
from blink.utils.annotate_video_frame import annotate_video_frame

WINDOW_SIZE = 13


def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_frame)


def detect(
    input_video: str,
    output_video: str = "output.mp4",
    max_height: int | None = None,
    start_frame: int = 0,
    should_rotate: bool = False
):
    left_blinks = 0
    right_blinks = 0
    full_blinks = 0

    annotations = Annotations("file.tag")

    cast_input_video = input_video
    try:
        cast_input_video = int(input_video)
    except ValueError:
        # Ignore the cast exception
        pass

    video = cv2.VideoCapture(cast_input_video)
    # video.open(input_video)

    if not video.isOpened():
        raise Exception(f"Unable to open video file {input_video}")

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)

    scale = 1.0
    if max_height and max_height < height:
        scale = max_height / height

    # orig = cv2.namedWindow("Original video", cv2.WINDOW_FULLSCREEN)

    window = FrameWindow(WINDOW_SIZE)

    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video, fourcc, fps, (int(width * scale), int(height * scale))
    )

    detector = LandmarkDetector()

    judge = SimpleJudge()

    frame_number = 0
    while video.isOpened():
        ret, frame = video.read()
        frame_number += 1

        if not ret:
            break

        cv2.resize(frame, (int(width * scale), int(height * scale)))


        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if frame_number < start_frame:
            continue

        # gray_frame = preprocess_frame(frame)
        # print(gray_frame)

        frame_data = FrameData(frame_number, frame)
        window.put(frame_data)

        annotated_frame = None
        if window.full():
            predictor.predict(window)

            judge.judge_blink(window)

            if window.cur.left_blink_state == BlinkState.start:
                left_blinks += 1
            if window.cur.right_blink_state == BlinkState.start:
                right_blinks += 1
            if window.cur.full_blink_state == BlinkState.start:
                full_blinks += 1

            window.cur.left_blinks = left_blinks
            window.cur.right_blinks = right_blinks
            window.cur.full_blinks = full_blinks
            # annotations.annotate_frame(cur)
            
            print(window.cur)
            annotations.annotate_frame(window.cur)
            annotated_frame = annotate_video_frame(window.cur)
            cv2.imshow("Original video", annotated_frame)

            if out:
                out.write(annotated_frame)
            # frame = annotate_video_frame(cur, frame)
        else:
            if out:
                out.write(frame)



        if cv2.waitKey(1) == "q":
            break
        else:
            print('write frame')
            # out.write(frame)

    video.release()
    if out:
        out.release()
