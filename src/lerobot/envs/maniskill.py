"""ManiskillEnv config for ``--env.type=maniskill``.

Wraps `ManiSkill 3 <https://maniskill.readthedocs.io>`_ tasks via lerobot's generic
gym-make path (in ``envs/factory.py``): when ``cfg.type`` is neither ``libero`` nor
``metaworld``, lerobot imports ``cfg.package_name`` (here ``mani_skill.envs`` — which
registers tasks like ``PickCube-v1``) and calls
``gym.make(cfg.gym_id, **cfg.gym_kwargs)``.

**Runtime caveats** (kept honest because we can't validate on macOS):

- ``mani_skill`` itself is intentionally NOT a transitive dep of lerobot. Install it on
  your Linux+CUDA box (``pip install --upgrade mani_skill torch``). macOS has no
  official ManiSkill support (the underlying SAPIEN package needs Vulkan + NVIDIA GPU),
  so on a Mac the schema/registration here loads fine but ``make_env`` will raise
  ``ModuleNotFoundError('mani_skill.envs')``.
- ManiSkill's raw obs is a nested dict (``agent.qpos``, ``sensor_data.<cam>.rgb``, …)
  that lerobot's ``preprocess_observation`` doesn't understand. End-to-end rollouts also
  require a ``ManiskillProcessorStep`` (mirroring ``IsaaclabArenaProcessorStep``); not
  shipped here because lerobot's ``make_env_pre_post_processors`` is hardcoded to
  ``libero``/``isaaclab_arena`` and we don't modify lerobot upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@EnvConfig.register_subclass("maniskill")
@dataclass
class ManiskillEnv(EnvConfig):
    """LeRobot env config wrapping a ManiSkill 3 task.

    Defaults target ``PickCube-v1`` with ``pd_ee_delta_pose`` (7-D delta EEF + gripper)
    and ``rgb`` obs mode (state + a single base camera). Customize for the task you load.
    """

    task: str | None = "PickCube-v1"  # ManiSkill gym id (no namespace prefix)
    fps: int = 20
    episode_length: int = 200

    # ManiSkill knobs ----------------------------------------------------------
    obs_mode: str = "rgb"
    # Default to a 7-D end-effector delta-pose action that mirrors LIBERO's action shape.
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "rgb_array"
    robot_uids: str = "panda"
    # Camera resolution requested via gym kwargs.
    sensor_height: int = 128
    sensor_width: int = 128

    # Optional extra gym kwargs for ManiSkill (num_envs, sim_backend, etc.).
    extra_gym_kwargs: dict[str, Any] = field(default_factory=dict)

    # Shape advertised to the policy. ManiSkill state vector size depends on the task and
    # robot; we expose it as a knob so users can match their task without touching code.
    # For the panda + pd_ee_delta_pose default, action_dim = 7 (xyz, axis-angle, gripper).
    action_dim: int = 7
    # State dim covering at least the robot proprio. ManiSkill's `obs["agent"]["qpos"]` is
    # 9-D for Panda by default (7 arm joints + 2 finger joints) — set explicitly so the
    # PolicyFeature shape matches the policy you intend to drive.
    state_dim: int = 9

    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Always advertise an ACTION + STATE feature; cameras are advertised when an
        # rgb-ish obs_mode is selected.
        if not self.features:
            self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
            self.features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))
            if self.obs_mode in ("rgb", "rgbd", "sensor_data", "rgbd+segmentation"):
                # Single base camera by default (matches ManiSkill's PickCube setup).
                self.features["base_camera"] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(self.sensor_height, self.sensor_width, 3),
                )
        if not self.features_map:
            self.features_map[ACTION] = ACTION
            self.features_map[OBS_STATE] = OBS_STATE
            if "base_camera" in self.features:
                self.features_map["base_camera"] = f"{OBS_IMAGES}.base_camera"

    # ------- lerobot factory hooks -------
    @property
    def package_name(self) -> str:
        # ManiSkill registers its gym ids on `import mani_skill.envs`.
        return "mani_skill.envs"

    @property
    def gym_id(self) -> str:
        # ManiSkill task ids are bare (e.g. "PickCube-v1"), with no namespace prefix.
        return self.task or "PickCube-v1"

    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "render_mode": self.render_mode,
            "robot_uids": self.robot_uids,
            "sensor_configs": {
                "height": self.sensor_height,
                "width": self.sensor_width,
            },
            "max_episode_steps": self.episode_length,
        }
        kwargs.update(self.extra_gym_kwargs)
        return kwargs
