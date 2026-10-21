import argparse
import platform

import torch
import torchvision

from fishcls.utils import get_device


def show_info(_: argparse.Namespace) -> None:
    print(f"python      {platform.python_version()}")
    print(f"torch       {torch.__version__}")
    print(f"torchvision {torchvision.__version__}")
    print(f"device      {get_device()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fishcls")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="versiones y dispositivo").set_defaults(run=show_info)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
