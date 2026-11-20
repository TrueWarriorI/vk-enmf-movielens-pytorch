"""Train and use a user-based ENMF recommender."""

import argparse
import json
from pathlib import Path
from zipfile import BadZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download MovieLens 1M from GroupLens")
    download.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))

    train = commands.add_parser("train", help="train ENMF and compare it with popularity")
    train.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    train.add_argument("--config", type=Path, default=Path("configs/movielens.json"))
    train.add_argument("--output", type=Path, required=True, help="new directory for this run")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    recommend = commands.add_parser("recommend", help="rank unseen items for one MovieLens user")
    recommend.add_argument("--checkpoint", type=Path, required=True)
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    try:
        if args.command == "download":
            from .data import download_movielens

            print(download_movielens(args.data))
        elif args.command == "train":
            from .config import TrainConfig
            from .training import train as run_training

            run_training(TrainConfig.read(args.config), args.data, args.output, args.device)
        else:
            from .recommend import recommend as get_recommendations

            if args.top_k < 1:
                parser.error("--top-k must be positive")
            print(json.dumps(get_recommendations(args.checkpoint, args.user_id, args.top_k), indent=2))
    except (ValueError, TypeError, OSError, BadZipFile) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
