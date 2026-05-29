"""Config for the web-button end-effector teleoperator."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("web_ee")
@dataclass
class WebEndEffectorTeleopConfig(TeleoperatorConfig):
    """Serves a tiny HTML control panel with directional + gripper buttons.

    The page sends press/release events for each axis; on every teleop tick the
    server snapshots the latest state into a ``keyboard_ee``-shaped action dict
    (``delta_x/y/z``, ``gripper``) so it can drive ``so100_follower_ee`` without
    keyboard access (e.g. when macOS Accessibility permission is unavailable).
    """

    host: str = "127.0.0.1"
    port: int = 8080
    use_gripper: bool = True
