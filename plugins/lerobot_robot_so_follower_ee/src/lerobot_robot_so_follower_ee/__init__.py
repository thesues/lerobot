"""Auto-discovery shim — registers the in-tree ``so_follower_ee`` robot type.

The real implementation lives as an additive lerobot module:
  - ``src/lerobot/robots/so_follower_ee/`` (registers draccus type
    ``"so100_follower_ee"`` / ``"so101_follower_ee"``)

This dist (``lerobot_robot_so_follower_ee``) satisfies the
``lerobot_robot_`` prefix that ``register_third_party_plugins()`` scans for,
so when ``lerobot-teleoperate main()`` calls the plugin registrar, this
``__init__.py`` runs and imports the real module — triggering registration.
"""

from lerobot.robots.so_follower_ee import (  # noqa: F401
    SO100FollowerEndEffector,
    SO100FollowerEndEffectorConfig,
    SOFollowerEndEffector,
    SOFollowerEndEffectorConfig,
)

__all__ = [
    "SO100FollowerEndEffector",
    "SO100FollowerEndEffectorConfig",
    "SOFollowerEndEffector",
    "SOFollowerEndEffectorConfig",
]
