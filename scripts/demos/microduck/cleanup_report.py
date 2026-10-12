# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a local review page for a measured MicroDuck cleanup recording."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def build_report(run_dir: Path, video: Path) -> Path:
    """Write a video review with measured phase, object status, and speeds [m/s]."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    samples = [
        {
            "t": row["time_s"],
            "phase": row["phase"],
            "command": row["command"][:3],
            "velocity": row.get("actual_velocity_m_s", [0, 0, 0]),
            "completed": row.get("objects_in_bin", row["completed"]),
            "assisted": row.get("sock_assistance_active", False),
        }
        for row in trace
    ]
    status = (
        "Both objects settled in the bin" if metrics["cleanup_complete"] else "Development run — cleanup incomplete"
    )
    pose = metrics.get("final_root_pose")
    upright = bool(pose and pose[2] > 0.075 and 1 - 2 * (pose[3] ** 2 + pose[4] ** 2) > 0.7)
    posture = "upright" if upright else "not upright"
    mode = metrics.get("parameters", {}).get("mode", "cleanup")
    if mode in {"marker", "sock"} and upright and metrics["objects_in_bin"].get(mode):
        status = f"{mode.capitalize()} placed in the bin — isolated task"
    runtime = f"{metrics['wall_seconds'] / 60:.1f} min wall time · {metrics['simulated_seconds']:.1f} s simulated"
    coupling_note = (
        "Diagnostic approximation: cloth forces were not applied back to the robot."
        if metrics.get("parameters", {}).get("coupling_relaxation", 1.0) == 0
        else "Robot and cloth exchange contact forces."
    )

    collision_note = (
        "Cloth contact with the beak is suppressed while attached, then restored after "
        f"{(metrics.get('assistance') or {}).get('release_collision_delay_s', 0):g} s of jaw opening. "
        "Other cloth contacts remain enabled. "
        if (metrics.get("assistance") or {}).get("mouth_collisions_suppressed_while_attached")
        else "Cloth contact with the robot remains enabled. "
    )
    seating_note = (
        "A 0.35 s assisted motion seats the patch between the partly closed jaws. "
        "Beak/cloth contact forces are not returned to the robot. "
        if (metrics.get("assistance") or {}).get("seating_duration_s")
        else ""
    )
    pinned_count = (metrics.get("assistance") or {}).get("maximum_pinned_nodes", 0)
    assistance_note = (
        f"ASSISTED DEMO: {pinned_count} cloth nodes were attached at the beak. "
        "The attachment releases when the jaw opens; its holding force is not returned to the robot. "
        f"{seating_note}{collision_note}Marker grasping remains contact-only."
        if metrics.get("assisted_grasp_enabled")
        else "Contact-only demo: no grasp attachment."
    )
    if metrics.get("learned_object_grasp_used"):
        initialization = metrics["policy_provenance"]["alpha_ground_pick"]["metadata"]["initialization"]
        grasp_note = (
            "Object-conditioned body and jaw grasping used a new random network with a geometric output bias. "
            if initialization == "random_geometric_output_bias"
            else "Object-conditioned body and jaw grasping continued from the Isaac-trained standing actor. "
        )
        policy_note = (
            "Standing and walking were trained from scratch in Isaac Lab + Newton. "
            + grasp_note
            + "No MuJoCo policy weights were used. Navigation, carrying head hold, "
            "and release sequencing remain scripted. "
            "No end-to-end cleanup policy was trained."
        )
    elif metrics.get("scratch_body_skills"):
        policy_note = (
            "Standing, reaching, and walking use Isaac Lab + Newton training from scratch. "
            "Jaw motion, head hold, and cleanup sequencing are scripted. No end-to-end cleanup policy was trained."
        )
    elif metrics.get("native_body_skills"):
        policy_note = (
            "Standing and pickup body skills were warm-started from MuJoCo "
            "and fine-tuned with PPO in Isaac Lab + Newton. "
            "Walking uses the existing Isaac Lab checkpoint. "
            "Jaw motion, head hold, and cleanup sequencing are scripted. "
            "No end-to-end cleanup policy was trained."
        )
    else:
        policy_note = (
            "Body motion uses released or overridden pickup skills and Isaac Lab-trained walking. "
            "Jaw motion, head hold, and cleanup sequencing are scripted. No end-to-end cleanup policy was trained."
        )
    bin_geometry = metrics.get("bin_geometry", {})
    basket_note = (
        f"Lidless wire basket: {100 * bin_geometry['height']:.1f} cm rim height. "
        "The wires and base have physical collisions; object containment uses its tapered interior."
        if bin_geometry.get("shape") == "tapered_wire"
        else ""
    )
    clearances = metrics.get("release_rim_clearance_m", {})
    if basket_note and clearances:
        basket_note += (
            " Measured clearance above the rim at release: "
            + ", ".join(f"{name} {100 * clearance:.1f} cm" for name, clearance in clearances.items())
            + "."
        )
    render_note = (
        "<p>RTX path-traced replay of the recorded Newton simulation. "
        '<a href="cleanup_render.usdc">Animated USD</a> · '
        '<a href="render_provenance.json">Rendering details</a></p>'
        if (run_dir / "cleanup_render.usdc").exists()
        else ""
    )
    video_url = html.escape(os.path.relpath(video.resolve(), run_dir.resolve()), quote=True)
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MicroDuck · Room cleanup</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111c24;color:#eaf0f3;font:16px system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:32px}h1{font-size:32px;margin:0 0 8px}
p{color:#b9c8cf;line-height:1.6}video{width:100%;border-radius:12px;background:#000}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.card{background:#20323e;padding:16px;border-radius:10px}.card small{color:#a5c0cb;display:block}
.card strong{display:block;font-size:22px;margin-top:8px}a{color:#6fe0d7}
details{margin-top:22px}img{width:100%;border-radius:12px}button,select{font:inherit;padding:6px 12px}
@media(max-width:700px){main{padding:16px}.cards{grid-template-columns:1fr 1fr}}
</style><main>
<h1>MicroDuck · Room cleanup</h1><p>__STATUS__ · Final robot posture: __POSTURE__</p>
<p>__RUNTIME__</p><p style="color:#ffd060"><strong>__ASSISTANCE__</strong></p>
<video id="video" controls playsinline src="__VIDEO__"></video>
<div class="cards">
<div class="card"><small>Current phase</small><strong id="phase">—</strong></div>
<div class="card"><small>Objects settled in bin</small><strong id="placed">0 / 2</strong></div>
<div class="card"><small>Commanded ground speed</small><strong id="command">0.000 m/s</strong></div>
<div class="card"><small>Measured body-plane speed</small><strong id="actual">0.000 m/s</strong></div>
</div>
<label>Playback speed <select id="rate">
<option>0.5</option><option selected>1</option><option>2</option></select> ×</label>
<p id="objects"></p><p id="assistance"></p>
<p>__BASKET__</p>__RENDER__
<details open><summary>Bedroom overview</summary>
<img src="room_overview.jpg" alt="Cutaway bedroom with bed, chair, MicroDuck, marker, fabric sock, and bin"></details>
<p>The sock is simulated cloth.
__POLICIES__
The standard setup uses low supports; this run does not demonstrate finding arbitrary objects on the floor.
__COUPLING__</p>
<p><a href="metrics.json">Measured results</a> · <a href="trace.jsonl">State trace</a> ·
<a href="scene.usda">Composed scene USD</a> · <a href="sock_nodes.npz">Cloth motion data</a></p>
</main><script>
const samples=__SAMPLES__,video=document.getElementById("video");
const show=(id,value)=>document.getElementById(id).textContent=value;
function update(){
 const row=samples[Math.min(samples.length-1,Math.round(video.currentTime/0.02))];
 if(row){
  show("phase",row.phase.replaceAll(":"," · ").replaceAll("_"," "));
  show("placed",Object.values(row.completed).filter(Boolean).length+" / 2");
  show("command",Math.hypot(...row.command.slice(0,2)).toFixed(3)+" m/s");
  show("actual",Math.hypot(...row.velocity.slice(0,2)).toFixed(3)+" m/s");
  show("assistance","Sock attachment: "+(row.assisted?"ACTIVE":"inactive"));
  show("objects","Marker: "+(row.completed.marker?"placed":"pending")
    +" · Sock: "+(row.completed.sock?"placed":"pending"));
 }
 requestAnimationFrame(update);
}
document.getElementById("rate").onchange=e=>video.playbackRate=Number(e.target.value);
update();
</script></html>
"""
    page = page.replace("__BASKET__", html.escape(basket_note)).replace("__RENDER__", render_note)
    page = page.replace("__POLICIES__", html.escape(policy_note))
    page = page.replace("__STATUS__", html.escape(status)).replace("__VIDEO__", video_url)
    page = page.replace("__POSTURE__", posture).replace("__RUNTIME__", runtime)
    page = page.replace("__COUPLING__", coupling_note).replace("__ASSISTANCE__", assistance_note)
    page = page.replace("__SAMPLES__", json.dumps(samples, separators=(",", ":")).replace("</", "<\\/"))
    destination = run_dir / "index.html"
    destination.write_text(page)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    print(build_report(args.run_dir, args.video or args.run_dir / "cleanup.mp4"))


if __name__ == "__main__":
    main()
