import argparse
import os

import typer
from typing_extensions import Annotated
from blink import detector
from blink.stats import accuracy_recall

from blink.trainer import nn
from blink.trainer import svm

app = typer.Typer()


@app.command()
def detect(
    input_video: Annotated[
        str, typer.Argument(help="Path to input video, or camera index")
    ],
    output_video: Annotated[
        str, typer.Argument(help="Path to output video")
    ] = "output.avi",
    data_path: Annotated[str | None, typer.Option(
        help="Path to opencv data")] = None,
    max_height: Annotated[int, typer.Option(
        help="Max height of output video")] = 1080,
    hidden: Annotated[int, typer.Option(help="Hide output window")] = 1080,
    start_frame: Annotated[int, typer.Option(help="Starting frame")] = 1,
):
    detector.detect(input_video, output_video, max_height, start_frame)


@app.command()
def stats(
    pred_tag_file: Annotated[
        str, typer.Argument(help="Path to the predicted tag file")
    ],
    ground_truth_tag_file: Annotated[
        str, typer.Argument(help="Path to the ground truth tag file")
    ],
):
    accuracy_recall(pred_tag_file, ground_truth_tag_file)


@app.command()
def calc_ear():
    trainer.calc_ear()


# def plot_ear():
#     trainer.plot_ear()
#

@app.command()
def train_nn(
    blink_ear_file: Annotated[str, typer.Argument(help="Path to the blink EAR file")],
    non_blink_ear_file: Annotated[
        str, typer.Argument(help="Path to the non blink EAR file")
    ],
    output_file: Annotated[str, typer.Argument(help="Neural network model file")],
):
    from blink.trainer import nn

    nn.run(blink_ear_file, non_blink_ear_file, output_file)


def train_svm():
    parser = argparse.ArgumentParser(description="Train SVM model")
    parser.add_argument("blink_ear", type=str,
                        help="Path to the blink EAR file")
    parser.add_argument(
        "non_blink_ear", type=str, help="Path to the non blink EAR file"
    )
    parser.add_argument(
        "output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)
                            ), "..", "models", "svm.xml"
        ),
        help="svm model file",
    )

    args = parser.parse_args()

    from blink.trainer import svm

    svm.run(args.blink_ear, args.non_blink_ear, args.output)


def main():
    app()
