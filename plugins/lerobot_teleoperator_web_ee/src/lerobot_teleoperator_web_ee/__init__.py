"""Auto-discovery shim — registers the in-tree ``web_ee`` teleoperator type."""

from lerobot.teleoperators.web_ee import (  # noqa: F401
    WebEndEffectorTeleop,
    WebEndEffectorTeleopConfig,
)

__all__ = ["WebEndEffectorTeleop", "WebEndEffectorTeleopConfig"]
