"""Config for the keyboard-EE driven single SO follower.

Wraps SOFollowerRobotConfig and adds the URDF + EE pipeline parameters needed
to translate end-effector delta commands (from `keyboard_ee` teleop) into
joint-space targets via inverse kinematics.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ..config import RobotConfig
from ..so_follower.config_so_follower import SOFollowerRobotConfig

# Bundled SO-ARM100 official URDF (so101_new_calib) + STL meshes. Pulled from
# https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101 so the
# plugin works out of the box without external clones.
_BUNDLED_URDF = str(Path(__file__).parent / "urdf" / "so101_new_calib.urdf")


def _default_step_sizes() -> dict[str, float]:
    return {"x": 0.02, "y": 0.02, "z": 0.02}


def _default_bounds() -> dict[str, list[float]]:
    return {"min": [-0.5, -0.5, -0.1], "max": [0.5, 0.5, 0.7]}


def _default_home_pose() -> dict[str, float]:
    # All arm joints at the calibrated middle (0°); gripper half-open.
    return {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    }


@RobotConfig.register_subclass("so101_follower_ee")
@RobotConfig.register_subclass("so100_follower_ee")
@dataclass
class SOFollowerEndEffectorConfig(SOFollowerRobotConfig):
    urdf_path: str = _BUNDLED_URDF
    target_frame_name: str = "gripper_frame_link"
    end_effector_step_sizes: dict[str, float] = field(default_factory=_default_step_sizes)
    end_effector_bounds: dict[str, list[float]] = field(default_factory=_default_bounds)
    # max EE step between consecutive frames (m); set well above per-tick step_size to disable the safety raise.
    max_ee_step_m: float = 1.0
    gripper_speed_factor: float = 20.0
    # IK weight knobs. SO-100 has only 5 DoF, so a 6-DoF orientation+position
    # target is over-constrained. With orientation_weight > 0 the solver fights
    # to preserve the current EE orientation, which manifests as wrist roll/flex
    # rotation coupled with position moves. Default to 0.0 (position-only IK)
    # so direction buttons produce clean linear EE motion.
    ik_position_weight: float = 1.0
    # 0.0 = pure position-only IK. Small positive values (e.g. 0.001) bias the
    # solver toward keeping EE orientation stable across ticks but can hurt
    # when 5 DoF aren't enough to satisfy both — leave at 0 unless tuning.
    ik_orientation_weight: float = 0.0
    # Per-tick joint-jog step (degrees) for the Roll button (web_ee roll →
    # wrist_roll). Roll is the only standalone rotational control; X/Y/Z position +
    # gripper pitch are handled by the 4-DoF IK. Hold the button to keep rotating.
    orientation_step_deg: float = 2.0
    # Elevation pitch (degrees) the IK holds the gripper at so it doesn't droop as
    # the arm moves. None → capture from the HOME pose on (re)connect / after HOME.
    # The bundled URDF's HOME elevation is ~0° (gripper pointing horizontally forward).
    hold_pitch_deg: float | None = None
    # Per-tick step for the direct per-motor joint jog (web_ee right-side panel:
    # ``joint_<motor>`` actions). Degrees for the arm joints; raw 0–100 units for the
    # gripper. This path bypasses the IK entirely — the recommended way to drive the
    # 5-DoF arm. Hold a button to keep jogging that motor.
    joint_jog_step_deg: float = 2.0
    # Hard cap on per-joint motion per tick (degrees), applied AFTER IK.
    # Only triggers when IK output jumps more than this between ticks (i.e.
    # the rare branch-flip near singularities); normal jogging is unaffected.
    # 10° @ 60 Hz = 600 °/s, well above servo speed, so this never slows
    # legitimate motion — it just clamps spikes. Set to 180 to disable.
    max_joint_step_deg: float = 10.0
    # Joint targets used when the teleop emits a one-shot ``{"home": True}`` action;
    # values are in the same units as ``SOFollower.send_action`` (degrees when
    # ``use_degrees=True``).
    home_pose: dict[str, float] = field(default_factory=_default_home_pose)


SO100FollowerEndEffectorConfig = SOFollowerEndEffectorConfig
SO101FollowerEndEffectorConfig = SOFollowerEndEffectorConfig
# Backward-compat alias (older docs/snippets referenced this name).
SOFollowerEndEffectorRobotConfig = SOFollowerEndEffectorConfig
