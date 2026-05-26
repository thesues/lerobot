"""``vllm`` policy: proxy action inference to a remote vLLM OpenPI policy server
(e.g. vllm-omni's GR00T-N1.7 deployment).

This subpackage lives alongside the built-in ``act/``, ``groot/``, ``pi0/``, etc., as an
additive lerobot module. The draccus type ``"vllm"`` is registered when this package is
imported, which is wired up via the ``lerobot_policy_vllm`` shim distribution under
``plugins/lerobot_policy_vllm/`` (auto-imported by lerobot's ``register_third_party_plugins()``).
"""

from .configuration_vllm import VllmConfig
from .modeling_vllm import VllmPolicy
from .processor_vllm import make_vllm_pre_post_processors

__all__ = ["VllmConfig", "VllmPolicy", "make_vllm_pre_post_processors"]
