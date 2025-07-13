#! /usr/bin/python3
import numpy as np
import typer

# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix

from blink.stats.eyeblink_entry import EyeBlinkEntry, MalformedEntryError

# Number of frames that base truth/prediction can differ and still count as a match
TOLLERANCE = 4


# import matplotlib.pyplot as plt
def __annotation_blink_state(file):
    with open(file) as f:
        lines = f.readlines()

    entries = []
    for line in lines:
        try:
            entry = EyeBlinkEntry()
            entry.parse_line(line)
            entries.append(entry)
        except MalformedEntryError:
            pass

    out = np.full(entries[-1].frame_id, False, dtype=bool)
    for entry in entries:
        if entry.blink_id >= 0:
            out[entry.frame_id] = True

    return out


def accuracy_recall(
    pred_file: str,
    base_file: str,
):
    pred_list = []
    base_list = []

    pred_list = __annotation_blink_state(pred_file)
    base_list = __annotation_blink_state(base_file)

    pred_out = []
    base_out = []
    i = 0
    while i < min(len(base_list), len(pred_list)):
        base_el = base_list[i]
        pred_el = pred_list[i]

        if not base_list[i - 1] and base_el:
            if base_el in pred_list[i - TOLLERANCE: i + TOLLERANCE]:
                base_out.append(1)
                pred_out.append(1)
                i += TOLLERANCE
            else:
                base_out.append(0)
                pred_out.append(1)

        if not pred_list[i - 1] and pred_el:
            if pred_el in base_list[i - TOLLERANCE: i + TOLLERANCE]:
                base_out.append(1)
                pred_out.append(1)
                i += TOLLERANCE
            else:
                base_out.append(1)
                pred_out.append(0)
        i += 1

    print(base_out)
    print(pred_out)
    cm = confusion_matrix(base_out, pred_out)
    print(cm)
    # color = "white"
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()

    report = classification_report(base_out, pred_out)
    print(report)
    #
    #
    # with open(base_file) as f:
    #     base = f.readlines()
    #
    # for i in range(0, min()window_size):
    #     local_entries.add(get_entry(local))
    #     base_entries.add(get_entry(base))
    #
    # while not (len(local) == 0 or len(base) == 0):
    #     if local_entries.cur_entry.frame_id == base_entries.cur_entry.frame_id:
    #         if base_entries.cur_blink:
    #             # print(f"{base_entries.cur_entry.frame_id}")
    #             # print(f"{base_entries.blink_in_window}")
    #             if local_entries.blink_in_window:
    #                 blink_success += 1
    #
    #             blink_total += 1
    #
    #             local_entries.add(get_entry(local))
    #             base_entries.add(get_entry(base))
    #
    #         if local_entries.cur_blink and not base_entries.blink_in_window:
    #             print(f"Error at frame: {local_entries.cur_entry.frame_id}")
    #
    #     elif local_entries.cur_entry.frame_id > base_entries.cur_entry.frame_id:
    #         base_entries.add(get_entry(base))
    #     else:
    #         local_entries.add(get_entry(local))
    #
    #     local_entries.add(get_entry(local))
    #     base_entries.add(get_entry(base))
    #
    # print(f"Blinks successfully detected: {blink_success}")
    # print(f"Accuracy: {blink_success / blink_total}")


# if __name__ == "__main__":
#     typer.run(main)
