"""Auto-discovery shim — registers the in-tree ``vllm`` policy and ``maniskill`` env.

The real implementations live as additive lerobot modules:
  - ``src/lerobot/policies/vllm/``  (registers draccus type ``"vllm"``)
  - ``src/lerobot/envs/maniskill.py`` (registers draccus type ``"maniskill"``)

Both modules register themselves at import time, but lerobot's
``register_third_party_plugins()`` only auto-imports installed dists whose name starts
with ``lerobot_robot_/camera_/teleoperator_/policy_``. This dist (``lerobot_policy_vllm``)
satisfies that prefix, so when ``lerobot-eval main()`` calls the plugin registrar, this
``__init__.py`` runs and imports the real modules — triggering both registrations.
"""

# Re-export the public API of the in-tree policy module (also serves as the registration trigger).
from lerobot.policies.vllm import VllmConfig, VllmPolicy, make_vllm_pre_post_processors  # noqa: F401

# Side-effect import: register the maniskill env alongside the policy.
try:
    from lerobot.envs import maniskill  # noqa: F401
except ImportError:
    # Maniskill registration is optional; if anything goes wrong with that module we
    # don't want to break vllm policy usage.
    pass

__all__ = ["VllmConfig", "VllmPolicy", "make_vllm_pre_post_processors"]
