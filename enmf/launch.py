"""Start a local CPU or GPU DDP run through torchrun."""

import argparse
import os
import subprocess
import sys
from uuid import uuid4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("training", nargs=argparse.REMAINDER, help="train followed by its options")
    args = parser.parse_args()
    if args.processes < 2 or not args.training or args.training[0] != "train":
        parser.error("Pass --processes >= 2 followed by train and its options.")

    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={args.processes}",
    ]
    env = os.environ.copy()
    if os.name == "nt":
        env["USE_LIBUV"] = "0"
        env["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
        env["PET_SIGNALS_TO_HANDLE"] = "SIGINT,SIGTERM"
        command.extend([
            "--rdzv-backend=c10d", "--rdzv-conf=store_type=file",
            f"--rdzv-id={uuid4()}", "--local-addr=127.0.0.1",
        ])
    else:
        command.append("--standalone")
    command.extend(["--module", "enmf", *args.training])
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
