"""Config for the joint-only web teleoperator (drives the plain so_follower)."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("web_so100")
@dataclass
class WebSO100TeleopConfig(TeleoperatorConfig):
    """Serves a tiny HTML panel with a per-motor +/- jog button for all 6 motors.

    Unlike ``web_ee`` there is NO inverse kinematics: the teleop tracks an
    absolute target position per motor (seeded from the robot's current pose) and
    each held button ramps that target at ``jog_speed_deg_s`` deg/s. ``get_action``
    returns ``{"<motor>.pos": target_deg}`` which drives the original
    ``so100_follower`` / ``so101_follower`` directly. Robot cameras (if any) are
    streamed live at the top of the same panel.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    # Jog speed in degrees per second while a button is held (fps-independent).
    jog_speed_deg_s: float = 30.0
