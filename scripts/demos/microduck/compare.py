# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record and compare native and transferred MicroDuck walking policies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write_comparison(output_dir: Path) -> Path:
    """Create synchronized videos, live telemetry, and a rollout comparison."""
    report = {name: json.loads((output_dir / f"{name}.json").read_text()) for name in ("native", "pretrained")}
    if report["native"]["command"] != report["pretrained"]["command"]:
        raise ValueError("The two recordings must use the same velocity command.")
    for name, result in report.items():
        if not (output_dir / f"{name}.mp4").is_file():
            raise FileNotFoundError(output_dir / f"{name}.mp4")
        if not result.get("playback_samples"):
            raise ValueError(f"{name}: re-record this rollout to collect synchronized velocity measurements.")
    template = Path(__file__).with_name("comparison.html").read_text()
    # Keep file paths and checkpoint names from ending the inline script element.
    data = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    destination = output_dir / "index.html"
    destination.write_text(template.replace("@@REPORT@@", data))
    return destination


def main() -> None:
    """Record both policies at the same command and write an offline comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("logs/microduck/comparison"))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--velocity", type=float, nargs=3, default=(0.15, 0.0, 0.0))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    demo = Path(__file__).resolve().parent
    common = ["--duration", str(args.duration), "--velocity", *map(str, args.velocity), "--viz", "newton_gl"]
    commands = [
        [sys.executable, str(demo / "evaluate.py"), "--checkpoint", str(args.checkpoint), "--num_envs", "1"],
        [sys.executable, str(demo / "play.py"), "--behavior", "alpha_walking"],
    ]

    def record(item: tuple[str, list[str]]) -> None:
        name, command = item
        with (args.output_dir / f"{name}.log").open("w") as log:
            subprocess.run(
                command
                + common
                + [
                    "--metrics",
                    str(args.output_dir / f"{name}.json"),
                    "--record",
                    str(args.output_dir / f"{name}.mp4"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(record, zip(("native", "pretrained"), commands)))
    print(write_comparison(args.output_dir).resolve())


if __name__ == "__main__":
    main()
