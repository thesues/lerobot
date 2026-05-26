# lerobot_policy_vllm (shim)

This is a **5-line auto-discovery shim**. It contains no functional code — it exists
solely because lerobot's `register_third_party_plugins()` (called at the top of
`lerobot-eval main()`) only auto-imports installed distributions whose name starts with
one of `lerobot_robot_/camera_/teleoperator_/policy_`.

The actual code lives where it belongs, as **additive lerobot modules**:

| Concept | File |
|---|---|
| `vllm` policy (registered as `--policy.type=vllm`) | `src/lerobot/policies/vllm/` |
| `maniskill` env (registered as `--env.type=maniskill`) | `src/lerobot/envs/maniskill.py` |

This shim's `__init__.py` simply imports those two modules, triggering their draccus
`register_subclass` decorators.

## Install

```bash
cd /Users/dongmao.zhang/upstream/lerobot
uv pip install -e plugins/lerobot_policy_vllm
```

That's it. After installing, run `lerobot-eval --help` and you'll see both `vllm` (under
`--policy.type`) and `maniskill` (under `--env.type`) listed.

## Why not just register from inside `src/lerobot/policies/vllm/__init__.py`?

Because `src/lerobot/policies/__init__.py` only explicitly imports the built-in policies
(`act`, `diffusion`, `groot`, …). To add `vllm` to that list we'd have to modify a
lerobot upstream file — which we deliberately don't do. The auto-discovery shim lets us
trigger our policy module's registration without touching any tracked lerobot file.
