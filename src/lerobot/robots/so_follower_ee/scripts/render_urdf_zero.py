"""Render the SO-101 URDF at its zero pose so you can see what to align the
physical arm to before running calibration.

Usage:
    .venv/bin/python -m lerobot.robots.so_follower_ee.scripts.render_urdf_zero
    # then either:
    rerun so101_zero_pose.rrd        # native viewer
    # or open the printed http://... URL if --serve is passed
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import placo
import rerun as rr
import trimesh

from scipy.spatial.transform import Rotation


def _origin_to_matrix(elem: ET.Element | None) -> np.ndarray:
    if elem is None:
        return np.eye(4)
    xyz = list(map(float, elem.get("xyz", "0 0 0").split()))
    rpy = list(map(float, elem.get("rpy", "0 0 0").split()))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--urdf",
        default=str(Path(__file__).resolve().parents[1] / "urdf" / "so101_new_calib.urdf"),
    )
    parser.add_argument("--out", default="so101_zero_pose.rrd", help="rerun .rrd file to write")
    parser.add_argument("--serve", action="store_true", help="start a web viewer instead of writing .rrd")
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    urdf_dir = urdf_path.parent

    # Parse URDF for the visual mesh of each link.
    tree = ET.parse(urdf_path)
    link_visuals: list[tuple[str, Path, np.ndarray]] = []
    for link in tree.getroot().findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        mesh_el = geom.find("mesh") if geom is not None else None
        if mesh_el is None:
            continue
        link_visuals.append((
            link.get("name"),
            (urdf_dir / mesh_el.get("filename")).resolve(),
            _origin_to_matrix(visual.find("origin")),
        ))

    # Compute every link's FK at the zero pose with placo.
    robot = placo.RobotWrapper(str(urdf_path))
    for j in robot.joint_names():
        robot.set_joint(j, 0.0)
    robot.update_kinematics()

    rr.init("so101_zero_pose")
    if args.serve:
        rr.serve_web(open_browser=True)
    else:
        rr.save(args.out)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Big base axes so the URDF +X / +Y / +Z directions are obvious.
    for axis, vec, color in [("x", [0.5, 0, 0], [255, 64, 64]), ("y", [0, 0.5, 0], [64, 255, 64]), ("z", [0, 0, 0.5], [64, 64, 255])]:
        rr.log(f"/world_axes/{axis}", rr.Arrows3D(vectors=[vec], origins=[[0, 0, 0]], colors=[color], radii=0.005), static=True)
        rr.log(f"/world_axes/{axis}_label", rr.Points3D(positions=[vec], colors=[color], radii=0.01, labels=[f"+{axis.upper()}"]), static=True)

    # Each link's mesh at its world pose.
    for link_name, mesh_path, T_link_visual in link_visuals:
        try:
            T_world_link = robot.get_T_world_frame(link_name)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {link_name}: {exc}", file=sys.stderr)
            continue
        if not mesh_path.exists():
            print(f"skip {link_name}: mesh not found at {mesh_path}", file=sys.stderr)
            continue
        mesh = trimesh.load(str(mesh_path), force="mesh")
        if not hasattr(mesh, "vertices"):
            continue
        T_total = T_world_link @ T_link_visual
        verts = np.asarray(mesh.vertices, dtype=float)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts_world = (T_total @ verts_h.T).T[:, :3]
        rr.log(
            f"/links/{link_name}",
            rr.Mesh3D(vertex_positions=verts_world, triangle_indices=np.asarray(mesh.faces, dtype=np.uint32)),
            static=True,
        )

    # End-effector marker + position printout.
    T_ee = robot.get_T_world_frame("gripper_frame_link")
    ee_pos = T_ee[:3, 3]
    rr.log(
        "/ee",
        rr.Points3D(positions=[ee_pos], colors=[[255, 200, 0]], radii=0.012, labels=[f"EE ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"]),
        static=True,
    )

    print(f"EE at zero pose: x={ee_pos[0]:.4f} m  y={ee_pos[1]:.4f} m  z={ee_pos[2]:.4f} m")
    if args.serve:
        print("Web viewer opened — keep this script running; Ctrl+C to exit.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print(f"Wrote {args.out}.  Open with:  rerun {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
