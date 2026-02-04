from blink.core.frame.data import FrameData
import cv2

from blink.core.state import EyeState


def annotate_video_frame(frame_data: FrameData):
    frame = frame_data.frame.copy()

    white = (255, 255, 255)
    blue = (255, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)

    font_face = cv2.FONT_HERSHEY_PLAIN
    font_thickness = 2
    font_height = 16
    line_height = font_height + 6
    font_size = cv2.getFontScaleFromHeight(
        font_face, font_height, font_thickness)

    h, w, c = frame.shape

    cv2.putText(
        frame,
        f"frame: {frame_data.frame_number}",
        (5, h - (line_height * 5) - 5),
        font_face,
        font_size,
        white,
        font_thickness,
    )
    cv2.putText(
        frame,
        f"time:            TODO",
        (5, h - (line_height * 4) - 5),
        font_face,
        font_size,
        white,
        font_thickness,
    )
    cv2.putText(
        frame,
        f"full blinks:       {frame_data.full_blinks}",
        (5, h - (line_height * 3) - 5),
        font_face,
        font_size,
        green,
        font_thickness,
    )
    cv2.putText(
        frame,
        f"blinks left:  {frame_data.left_blinks}",
        (5, h - (line_height * 2) - 5),
        font_face,
        font_size,
        green,
        font_thickness,
    )
    cv2.putText(
        frame,
        f"blinks right: {frame_data.right_blinks}",
        (5, h - (line_height * 1) - 5),
        font_face,
        font_size,
        green,
        font_thickness,
    )

    if frame_data.left_eye_bbox and frame_data.left_eye_state == EyeState.closed:
        lt, lb = frame_data.left_eye_bbox
        color = red

        lt_pad = [sum(x) for x in zip(lt , (-15, -15))]
        lb_pad = [sum(x) for x in zip(lb , (15, 15))]

        cv2.rectangle(frame, lt_pad, lb_pad, color, 1)
    
    if frame_data.right_eye_bbox and frame_data.right_eye_state == EyeState.closed:
        rt, rb = frame_data.right_eye_bbox
        color = red
        
        rt_pad = [sum(x) for x in zip(rt , (-15, -15))]
        rb_pad = [sum(x) for x in zip(rb , (15, 15))]
        cv2.rectangle(frame, rt_pad, rb_pad, color, 1)

     #     cv::rectangle(
     #         frame, *data->data()->getFaceTopLeft(), cv::Scalar(0, 255, 0), 3);
     # }

    # cv2.putText(frame, f"blinks/min: " + std::to_string(data->meta()->blinksFull() /
    #                                             (data->meta()->getTime() / 60)),
    #
    #             cv::Point(5, data->frame()->rows - 5), cv::FONT_HERSHEY_PLAIN, 1.5,
    #             white, 2);
    #
    #     cv::putText(frame,
#         "blink left: " + std::to_string(data->meta()->blinksLeft()),
#         cv::Point(data->frame()->cols - 200,
#             data->frame()->rows - (line_height * 3) - 5),
#         font_face, font_size, green, font_thickness);
#     cv::putText(frame,
#         "blink right: " + std::to_string(data->meta()->blinksRight()),
#         cv::Point(data->frame()->cols - 200,
#             data->frame()->rows - (line_height * 2) - 5),
#         font_face, 1.5, blue, font_thickness);
#     cv::putText(frame,
#         "blink total: " + std::to_string(data->meta()->blinksFull()),
#         cv::Point(data->frame()->cols - 200,
#             data->frame()->rows - (line_height * 1) - 5),
#         font_face, font_size, white, font_thickness);

    return frame


# {
#     if (data->data()->getFaceTopLeft() &&
#         (data->meta()->blinkStateLeft() == BlinkState::start ||
#             data->meta()->blinkStateLeft() == BlinkState::continued))
#     {
#         cv::rectangle(
#             frame, *data->data()->getFaceTopLeft(), cv::Scalar(0, 255, 0), 3);
#     }
#     if (data->data()->getFaceTopRight() &&
#         (data->meta()->blinkStateRight() == BlinkState::start ||
#             data->meta()->blinkStateRight() == BlinkState::continued))
#     {
#         cv::rectangle(
#             frame, *data->data()->getFaceTopRight(), cv::Scalar(255, 0, 0), 3);
#     }
#
#     int font_face = cv::FONT_HERSHEY_PLAIN;
#     int font_thickness = 2;
#     int font_height = 16;
#     int line_height = font_height + 6;
#     double font_size =
#         cv::getFontScaleFromHeight(font_face, font_height, font_thickness);
#
#     cv::Scalar white = cv::Scalar(255, 255, 255);
#     cv::Scalar blue = cv::Scalar(255, 0, 0);
#     cv::Scalar green = cv::Scalar(0, 255, 0);
#
#     cv::putText(frame,
#         "frame: " + std::to_string(data->meta()->getFrameNumber()),
#         cv::Point(5, data->frame()->rows - (line_height * 5) - 5), font_face,
#         font_size, white, font_thickness);
#     cv::putText(frame, "time: " + std::to_string(data->meta()->getTime()),
#         cv::Point(5, data->frame()->rows - (line_height * 4) - 5), font_face,
#         font_size, white, font_thickness);
#     cv::putText(frame,
#         "eye state:      " + std::to_string(data->meta()->eyesState()),
#         cv::Point(5, data->frame()->rows - (line_height * 3) - 5), font_face,
#         font_size, white, font_thickness);
#     cv::putText(frame,
#         "eye state left: " + std::to_string(data->meta()->eyeStateLeft()),
#         cv::Point(5, data->frame()->rows - (line_height * 2) - 5), font_face,
#         font_size, white, font_thickness);
#     cv::putText(frame,
#         "eye state right: " + std::to_string(data->meta()->eyeStateRight()),
#         cv::Point(5, data->frame()->rows - (line_height * 1) - 5), font_face,
#         font_size, white, font_thickness);
#
#     cv::putText(frame,
#         "blink left: " + std::to_string(data->meta()->blinksLeft()),
#         cv::Point(data->frame()->cols - 200,
#             data->frame()->rows - (line_height * 3) - 5),
#         font_face, font_size, green, font_thickness);
#     cv::putText(frame,
#         "blink right: " + std::to_string(data->meta()->blinksRight()),
#         cv::Point(data->frame()->cols - 200,
#             data->frame()->rows - (line_height * 2) - 5),
#         font_face, 1.5, blue, font_thickness);
#     cv::putText(frame,
#         "blink total: " + std::to_string(data->meta()->blinksFull()),
#         cv::Point(data->frame()->cols - 200,
#             data->frame()->rows - (line_height * 1) - 5),
#         font_face, font_size, white, font_thickness);
#
#     if (data->meta()->getTime() > 0)
#     {
#         cv::putText(frame,
#             "blinks/min: " + std::to_string(data->meta()->blinksFull() /
#                                             (data->meta()->getTime() / 60)),
#
#             cv::Point(5, data->frame()->rows - 5), cv::FONT_HERSHEY_PLAIN, 1.5,
#             white, 2);
#     }
#
# #ifdef _DEBUG
#     if (data->data()->getFace())
#     {
#         cv::rectangle(
#             frame, *data->data()->getFace(), cv::Scalar(0, 0, 255), 3);
#     }
#     if (data->data()->getFaceTopLeft())
#     {
#         cv::rectangle(
#             frame, *data->data()->getFaceTopLeft(), cv::Scalar(255, 0, 255), 3);
#     }
#     if (data->data()->getFaceTopRight())
#     {
#         cv::rectangle(frame, *data->data()->getFaceTopRight(),
#             cv::Scalar(0, 255, 255), 3);
#     }
# #endif
# }
