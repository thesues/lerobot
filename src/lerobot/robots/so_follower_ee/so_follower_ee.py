"""Single SO follower driven by keyboard end-effector deltas.

This robot wraps :class:`SOFollower` and accepts the action schema produced by
:class:`KeyboardEndEffectorTeleop` (``delta_x/delta_y/delta_z/gripper``). On
``send_action`` it runs an FK→delta-apply→IK pipeline to translate the EE
deltas into joint targets, then forwards joint commands to the underlying
follower bus.

Designed so that ``lerobot-teleoperate --robot.type=so100_follower_ee
--teleop.type=keyboard_ee`` works end-to-end without modifying upstream
lerobot files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    TransitionKey,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.utils.decorators import check_if_not_connected

from ..robot import Robot
from ..so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.rotation import Rotation
from ..so_follower.so_follower import SOFollower
from .config_so_follower_ee import SOFollowerEndEffectorConfig

logger = logging.getLogger(__name__)

# Arm joints used by the kinematic chain. The gripper is handled separately
# by GripperVelocityToJoint and is the 6th motor of the SO follower.
ARM_JOINTS: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Joints driven by the IK. The SO-100/101 is a 5-DoF arm; we treat 4 of its joints
# as a "4-DoF + EE" group solved by IK to reach an end-effector target of
# (x, y, z, pitch). Picking exactly 4 joints for a 4-D task keeps the task Jacobian
# square (4-D target → 4 joints → unique solution), so there's no null-space
# ambiguity to flip through tick-to-tick. wrist_roll is NOT in the IK group — it's
# a pure single-joint jog (roll), and the gripper is handled separately.
IK_FREE_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
)

# Gripper local axis that "points out" of the gripper (toward the grasp). From the
# bundled URDF this is local +Z (world (1,0,0) at HOME — horizontal, forward). Its
# elevation angle asin(R @ axis)[2] is the EE pitch the IK holds steady.
EE_POINTING_AXIS_LOCAL: tuple[float, float, float] = (0.0, 0.0, 1.0)

# Joints held fixed by the (currently unused) placo-based WeightedInverseKinematics
# step. Defined here so that class doesn't NameError if it's ever wired back in.
FROZEN_JOINTS: tuple[str, ...] = ()

# Orientation jog mapping. The 5-DoF arm physically cannot track an arbitrary EE
# orientation, and yaw/pitch are absorbed into the 4-DoF IK (yaw is implicit in the
# X/Y position target; pitch is auto-held by the IK). The only standalone rotational
# control is roll → wrist_roll (spin about the gripper axis), a pure single-joint jog.
ORIENTATION_JOG_JOINTS: dict[str, str] = {
    "delta_roll": "wrist_roll",
}


@dataclass
class JointDeltaLimit(RobotActionProcessorStep):
    """Cap per-joint motion per tick to absorb IK oscillations and elbow flips.

    Runs after IK. Maintains the previous tick's command per joint and clips
    each new command to ``prev ± max_step_deg``. ``reset()`` clears state so
    the next tick after a HOME jump or pipeline restart doesn't artificially
    limit the recovery.
    """

    max_step_deg: float = 5.0
    _prev: dict[str, float] | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if self._prev is None:
            self._prev = {k: float(v) for k, v in action.items() if k.endswith(".pos")}
            return action
        prev = self._prev
        capped: dict[str, float] = {}
        for k, v in action.items():
            if not k.endswith(".pos"):
                capped[k] = v if not isinstance(v, float) else float(v)
                continue
            new = float(v)
            p = prev.get(k)
            if p is not None:
                delta = new - p
                if delta > self.max_step_deg:
                    new = p + self.max_step_deg
                elif delta < -self.max_step_deg:
                    new = p - self.max_step_deg
            capped[k] = new
        self._prev = {k: float(v) for k, v in capped.items() if k.endswith(".pos")}
        return capped

    def transform_features(self, features):
        return features

    def reset(self) -> None:
        self._prev = None


@dataclass
class PositionPitchHoldIK(RobotActionProcessorStep):
    """4-DoF Jacobian IK (position + held elevation pitch) with damped least squares.

    Solves the four IK joints (``IK_FREE_JOINTS`` =
    shoulder_pan/shoulder_lift/elbow_flex/wrist_flex) for a 4-D end-effector task:
    the (x, y, z) position from the teleop deltas PLUS the gripper's elevation
    pitch, which is held at a fixed reference so the gripper does not droop as the
    arm reconfigures.

        Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e      e = [p_target − p_curr ; w·(θ_hold − θ_curr)]

    where J (4×4) stacks the 3-row position Jacobian and a 1-row elevation Jacobian.
    Since the task is 4-D and exactly 4 joints are free, J is square and the
    solution is unique (modulo damping near singular configs) — continuous in the
    input, so no twitching.

    Elevation pitch θ = asin((R · pointing_local)_z) is azimuth-independent
    (shoulder_pan rotates about world Z, preserving z-components), so holding it is
    unaffected by X/Y moves that swing the base. The hold target is captured from
    the first observation after ``reset()`` (i.e. the HOME pose), or fixed via
    ``hold_pitch_rad``.

    Orientation columns of the action (``ee.wx/wy/wz``) are popped but ignored —
    EE yaw/roll are not tracked here (yaw is implicit in the X/Y target; roll is a
    standalone wrist_roll jog applied downstream by ``OrientationJointJog``).
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    free_joint_names: tuple[str, ...] = IK_FREE_JOINTS
    pointing_axis_local: tuple[float, float, float] = EE_POINTING_AXIS_LOCAL
    damping: float = 0.05
    gain: float = 1.0
    pitch_weight: float = 1.0
    # Fixed elevation pitch to hold (radians). None → capture from first post-reset
    # observation (the HOME pose).
    hold_pitch_rad: float | None = None
    _pitch_target: float | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = float(action.pop("ee.x"))
        y = float(action.pop("ee.y"))
        z = float(action.pop("ee.z"))
        action.pop("ee.wx", None)
        action.pop("ee.wy", None)
        action.pop("ee.wz", None)
        gripper_pos = float(action.pop("ee.gripper_pos"))

        observation = self.transition.get(TransitionKey.OBSERVATION)
        arm_motors = [m for m in self.motor_names if m != "gripper"]
        q_obs_deg = np.array(
            [float(observation[f"{m}.pos"]) for m in arm_motors], dtype=float
        )

        # Sync placo's internal robot state to the observation, then read the EE
        # pose + Jacobian for the free joints at this configuration.
        robot = self.kinematics.robot
        for joint_name, q_deg in zip(arm_motors, q_obs_deg, strict=True):
            robot.set_joint(joint_name, float(np.deg2rad(q_deg)))
        robot.update_kinematics()
        t_curr = robot.get_T_world_frame(self.kinematics.target_frame_name)
        p_curr = t_curr[:3, 3]
        pointing = t_curr[:3, :3] @ np.asarray(self.pointing_axis_local, dtype=float)
        pz = float(np.clip(pointing[2], -1.0, 1.0))
        elev_curr = float(np.arcsin(pz))
        if self._pitch_target is None:
            self._pitch_target = (
                self.hold_pitch_rad if self.hold_pitch_rad is not None else elev_curr
            )

        J = robot.frame_jacobian(self.kinematics.target_frame_name, "local_world_aligned")
        free_cols = [robot.get_joint_v_offset(j) for j in self.free_joint_names]
        J_pos = J[:3, free_cols]  # 3 × 4
        J_ang = J[3:6, free_cols]  # 3 × 4

        # d(elevation)/dq_i = d(asin(pz))/dq_i, with d(pointing)/dq_i = ω_i × pointing.
        d_pointing = np.cross(J_ang.T, pointing)  # (4, 3): ω_i × pointing per joint
        denom = float(np.sqrt(max(1.0 - pz * pz, 1e-6)))
        J_elev = (d_pointing[:, 2] / denom).reshape(1, -1)  # 1 × 4

        err_pos = np.array([x - p_curr[0], y - p_curr[1], z - p_curr[2]], dtype=float)
        err_pitch = self._pitch_target - elev_curr

        J_task = np.vstack([J_pos, self.pitch_weight * J_elev])  # 4 × 4
        err = np.concatenate([err_pos, [self.pitch_weight * err_pitch]])  # (4,)

        # Damped least squares: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err
        lam2 = self.damping * self.damping
        n = J_task.shape[0]
        dq_rad = J_task.T @ np.linalg.solve(J_task @ J_task.T + lam2 * np.eye(n), err)
        dq_deg = np.rad2deg(dq_rad) * self.gain

        q_target_deg = q_obs_deg.copy()
        for k, joint_name in enumerate(self.free_joint_names):
            idx = arm_motors.index(joint_name)
            q_target_deg[idx] = q_obs_deg[idx] + dq_deg[k]

        for i, m in enumerate(self.motor_names):
            if m == "gripper":
                action["gripper.pos"] = gripper_pos
            else:
                action[f"{m}.pos"] = float(q_target_deg[i])
        return action

    def transform_features(self, features):
        return features

    def reset(self) -> None:
        # Re-capture the hold pitch from the next observation (after a HOME jump).
        self._pitch_target = None


@dataclass
class WeightedInverseKinematics(InverseKinematicsEEToJoints):
    """IK step that forwards configurable position/orientation weights to placo.

    The base ``InverseKinematicsEEToJoints`` always calls
    ``RobotKinematics.inverse_kinematics`` with default weights (1.0 / 0.01).
    On a 5-DoF SO-100 the orientation task is over-constrained relative to the
    arm's DoF and forces wrist roll/flex to fight, so we expose the weights
    here.
    """

    position_weight: float = 1.0
    orientation_weight: float = 0.0

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        q_raw = np.array(
            [
                float(v)
                for k, v in observation.items()
                if isinstance(k, str) and k.endswith(".pos")
            ],
            dtype=float,
        )

        if self.initial_guess_current_joints or self.q_curr is None:
            self.q_curr = q_raw

        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_target = self.kinematics.inverse_kinematics(
            self.q_curr,
            t_des,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
        )
        self.q_curr = q_target

        # Index of each arm joint inside ``motor_names`` and ``q_target``.
        # The kinematic chain (``ARM_JOINTS``) is a prefix of ``motor_names``
        # in this robot, so the indices line up 1:1.
        for i, name in enumerate(self.motor_names):
            if name == "gripper":
                action["gripper.pos"] = float(gripper_pos)
            elif name in FROZEN_JOINTS:
                # Hold the joint at its observed value — removing it from the
                # IK DoF eliminates the null-space jitter that placo's QP picks
                # arbitrarily each tick on a redundant 5-DoF arm.
                action[f"{name}.pos"] = float(q_raw[i])
            else:
                action[f"{name}.pos"] = float(q_target[i])
        return action


@dataclass
class KeyboardDeltaToEECmd(RobotActionProcessorStep):
    """Bridge KeyboardEndEffectorTeleop / web_ee output → EEReferenceAndDelta input.

    Position deltas (delta_x/y/z) feed the EE-pose pipeline. The EE pitch is held
    by the IK (not commanded here) and EE yaw is implicit in the X/Y target, so the
    IK orientation target stays identity. The only standalone rotational control is
    roll: ``delta_roll`` is passed through untouched as ``jog_roll`` for
    ``OrientationJointJog`` to apply as a direct wrist_roll jog after the IK.
    """

    def action(self, action: RobotAction) -> RobotAction:
        dx = float(action.pop("delta_x", 0.0))
        dy = float(action.pop("delta_y", 0.0))
        dz = float(action.pop("delta_z", 0.0))
        # KeyboardEndEffectorTeleop emits gripper ∈ {0, 1, 2}; defaults to 1 (stay).
        gripper = float(action.pop("gripper", 1.0))
        action["enabled"] = bool(dx or dy or dz)
        action["target_x"] = dx
        action["target_y"] = dy
        action["target_z"] = dz
        action["target_wx"] = 0.0
        action["target_wy"] = 0.0
        action["target_wz"] = 0.0
        action["gripper_vel"] = gripper
        # Carry the roll jog direction (−1/0/+1) through to OrientationJointJog.
        action["jog_roll"] = float(action.pop("delta_roll", 0.0))
        return action

    def transform_features(self, features):
        return features


@dataclass
class OrientationJointJog(RobotActionProcessorStep):
    """Apply the roll jog direction as a direct wrist_roll offset, after the IK.

    Runs once the IK has written the ``{joint}.pos`` targets. If the roll jog
    direction is non-zero, it adds ``dir * step_deg`` to wrist_roll's target.
    wrist_roll is not in the IK group (it's held at the observed value by the IK),
    so layering the jog on top integrates smoothly tick-to-tick via the real motor
    feedback. Rolling about the gripper's pointing axis does not change its
    elevation, so it never disturbs the IK's pitch hold (see ``ORIENTATION_JOG_JOINTS``).
    """

    step_deg: float = 2.0
    jog_joints: dict[str, str] = field(default_factory=lambda: dict(ORIENTATION_JOG_JOINTS))

    def action(self, action: RobotAction) -> RobotAction:
        for jog_key, joint in self.jog_joints.items():
            # Key name in the action is the jog_* form produced by KeyboardDeltaToEECmd.
            direction = float(action.pop(jog_key.replace("delta_", "jog_"), 0.0))
            if direction == 0.0:
                continue
            pos_key = f"{joint}.pos"
            if pos_key in action:
                action[pos_key] = float(action[pos_key]) + direction * self.step_deg
        return action

    def transform_features(self, features):
        return features

    def reset(self) -> None:
        return None


class SOFollowerEndEffector(Robot):
    """SO follower that converts EE-delta actions to joint commands via IK."""

    config_class = SOFollowerEndEffectorConfig
    name = "so100_follower_ee"

    def __init__(self, config: SOFollowerEndEffectorConfig):
        super().__init__(config)
        self.config = config
        self._inner = SOFollower(config)
        self._pipeline: RobotProcessorPipeline | None = None
        # Cached previous joint command — replayed verbatim when the teleop is
        # idle so we don't re-run IK on noisy observations and twitch the motors.
        self._last_joint_action: dict[str, float] | None = None

    def _build_pipeline(self) -> RobotProcessorPipeline:
        kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
            joint_names=ARM_JOINTS,
        )
        motor_names = list(self._inner.bus.motors.keys())
        return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[
                KeyboardDeltaToEECmd(),
                EEReferenceAndDelta(
                    kinematics=kinematics,
                    end_effector_step_sizes=self.config.end_effector_step_sizes,
                    motor_names=ARM_JOINTS,
                    use_latched_reference=False,
                ),
                EEBoundsAndSafety(
                    end_effector_bounds=self.config.end_effector_bounds,
                    max_ee_step_m=self.config.max_ee_step_m,
                ),
                GripperVelocityToJoint(
                    speed_factor=self.config.gripper_speed_factor,
                    discrete_gripper=True,
                ),
                PositionPitchHoldIK(
                    kinematics=kinematics,
                    motor_names=motor_names,
                    hold_pitch_rad=(
                        None
                        if self.config.hold_pitch_deg is None
                        else float(np.deg2rad(self.config.hold_pitch_deg))
                    ),
                ),
                OrientationJointJog(step_deg=self.config.orientation_step_deg),
                JointDeltaLimit(max_step_deg=self.config.max_joint_step_deg),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        return self._inner.observation_features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_roll": float,
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._inner.connect(calibrate=calibrate)
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self._inner.is_calibrated

    def calibrate(self) -> None:
        self._inner.calibrate()

    def configure(self) -> None:
        self._inner.configure()

    def get_observation(self) -> RobotObservation:
        return self._inner.get_observation()

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if action.get("home"):
            # Bypass IK and write the configured home joint pose directly. Reset
            # the pipeline so that the next jog re-latches the new EE reference.
            home_action = {f"{motor}.pos": float(val) for motor, val in self.config.home_pose.items()}
            result = self._inner.send_action(home_action)
            if self._pipeline is not None:
                self._pipeline.reset()
            self._last_joint_action = dict(home_action)
            return result

        is_idle = (
            float(action.get("delta_x", 0.0)) == 0.0
            and float(action.get("delta_y", 0.0)) == 0.0
            and float(action.get("delta_z", 0.0)) == 0.0
            and float(action.get("delta_roll", 0.0)) == 0.0
            and float(action.get("gripper", 1.0)) == 1.0
        )
        if is_idle and self._last_joint_action is not None:
            # Re-send the last commanded joints so motors hold their target;
            # skipping IK avoids re-solving from noisy observations every tick.
            return self._inner.send_action(self._last_joint_action)

        obs = self._inner.get_observation()
        joint_action = self._pipeline((dict(action), obs))
        self._last_joint_action = dict(joint_action)
        return self._inner.send_action(joint_action)

    def disconnect(self) -> None:
        self._inner.disconnect()
        logger.info(f"{self} disconnected.")


SO100FollowerEndEffector = SOFollowerEndEffector
SO101FollowerEndEffector = SOFollowerEndEffector
