# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show the released MicroDuck policies consecutively, optionally recording each."""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from pathlib import Path

from behaviors import BEHAVIORS


def write_gallery(output_dir: Path, behaviors: list[str]) -> None:
    """Write an offline video gallery beside the recorded MP4 files."""
    labels = {
        "alpha_walking": "Walking",
        "velstand": "Walking and standing",
        "alpha_stand": "Getting up from the floor",
        "alpha_sitstand": "Sit and stand",
        "alpha_ground_pick": "Ground pick",
        "ball_kick_left": "Left-foot kick",
        "ball_kick_right": "Right-foot kick",
        "roulade": "Forward roll",
        "roller": "Roller skating",
        "roller_crouch": "Crouch on rollers",
    }
    cards = "".join(
        f"<article><h2>{html.escape(labels[name])}</h2>"
        f'<video controls preload="metadata" src="{html.escape(name)}.mp4"></video></article>'
        for name in behaviors
    )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MicroDuck policy showcase</title>
<style>
body{font:17px system-ui;background:#101b26;color:#eef4fb;margin:32px auto;max-width:1200px;padding:0 20px}
h1{font-size:30px}h2{font-size:19px}p{color:#b9c9d8}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}
article{background:#1b2a38;border-radius:12px;padding:16px}video{width:100%;border-radius:8px}
</style><h1>MicroDuck policy showcase</h1><p>Trained policies running in Isaac Lab + Newton.
Select a clip to watch its behavior.</p><main>"""
        + cards
        + """</main>
<script>document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>{
document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause()})}));</script></html>"""
    )
    (output_dir / "index.html").write_text(page)


def main() -> None:
    """Launch one isolated Isaac Lab scene per selected behavior."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behaviors", choices=BEHAVIORS, nargs="+", default=list(BEHAVIORS))
    parser.add_argument("--asset_dir", type=Path, default=Path("logs/microduck/assets"))
    parser.add_argument("--output_dir", type=Path, default=Path("logs/microduck/showcase"))
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per behavior [s].")
    parser.add_argument("--viz", choices=("newton_gl", "none"), default="newton_gl")
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--record", action="store_true", help="Save one MP4 per behavior; requires the video extra.")
    args = parser.parse_args()
    if not 0 < args.duration < float("inf"):
        parser.error("--duration must be finite and positive.")
    if args.record and args.viz == "none":
        parser.error("--record requires --viz newton_gl.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.behaviors:
        command = [
            sys.executable,
            str(Path(__file__).with_name("play.py")),
            "--behavior",
            name,
            "--asset_dir",
            str(args.asset_dir),
            "--duration",
            str(args.duration),
            "--viz",
            args.viz,
            "--metrics",
            str(args.output_dir / f"{name}.json"),
            "--trace",
            str(args.output_dir / f"{name}.jsonl"),
        ]
        if name == "alpha_stand":
            command += ["--start_pose", "prone"]
        if args.real_time:
            command.append("--real_time")
        if args.record:
            command += ["--record", str(args.output_dir / f"{name}.mp4")]
        print(f"Showing {name}", flush=True)
        subprocess.run(command, check=True)
    if args.record:
        write_gallery(args.output_dir, args.behaviors)


if __name__ == "__main__":
    main()
