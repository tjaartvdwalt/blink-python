from blink.core.frame.window import FrameWindow
from blink.core.state import BlinkState, EyeState


class SimpleJudge:
    def judge_blink(self, window: FrameWindow):
        prev = window.prev
        cur = window.cur
        print(cur)

        # Left eye blink
        if (
            prev.left_eye_state != EyeState.closed
            and cur.left_eye_state == EyeState.closed
        ):
            cur.left_blink_state = BlinkState.start
        elif (
            prev.left_eye_state == EyeState.closed
            and cur.left_eye_state == EyeState.closed
        ):
            cur.left_blink_state = BlinkState.continued
        else:
            cur.left_blink_state == BlinkState.no

        # Right eye blink
        if (
            prev.right_eye_state != EyeState.closed
            and cur.right_eye_state == EyeState.closed
        ):
            cur.right_blink_state = BlinkState.start
        elif (
            prev.right_eye_state == EyeState.closed
            and cur.right_eye_state == EyeState.closed
        ):
            cur.right_blink_state = BlinkState.continued
        else:
            cur.right_blink_state == BlinkState.no

        # Full blink
        if (
            cur.left_blink_state == BlinkState.start
            or cur.left_blink_state == BlinkState.continued
        ) and (
            cur.right_blink_state == BlinkState.start
            or cur.right_blink_state == BlinkState.continued
        ):
            if prev.full_blink_state == BlinkState.no:
                cur.full_blink_state = BlinkState.start
            else:
                cur.full_blink_state = BlinkState.continued
        else:
            cur.full_blink_state = BlinkState.no
