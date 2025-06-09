import argparse
import os

def inference():
    parser = argparse.ArgumentParser(description="Say hi.")
    parser.add_argument("target", type=str, help="the name of the target")
    parser.add_argument(
        "--end",
        dest="end",
        default="!",
        help="sum the integers (default: find the max)",
    )

    args = parser.parse_args()

    from .main import main

    main()(args.target, end=args.end)


def calc_ear():
    trainer.calc_ear()


def plot_ear():
    trainer.plot_ear()

def train_nn():
    parser = argparse.ArgumentParser(description="Train neural network")
    parser.add_argument("blink_ear", type=str,
                        help="Path to the blink EAR file")
    parser.add_argument(
        "non_blink_ear", type=str, help="Path to the non blink EAR file"
    )
    parser.add_argument(
        "output",
        type=str,
        help="neural network model file",
    )

    args = parser.parse_args()

    from blink.trainer import nn

    nn.run(args.blink_ear, args.non_blink_ear, args.output)

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
