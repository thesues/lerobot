# examples/vllm_policy

A ready-to-use `--policy.path` directory for the `vllm` policy (remote vLLM OpenPI
server), plus helper scripts for regenerating the config and running a single-task
visualized rollout.

The actual policy code lives at `src/lerobot/policies/vllm/`. The ManiSkill env (also
registered by the same plugin shim) lives at `src/lerobot/envs/maniskill.py`. The shim
distribution that wires them into lerobot's auto-discovery is
`plugins/lerobot_policy_vllm/`.

## Quick start

```bash
cd /Users/dongmao.zhang/upstream/lerobot

# Install the shim once (also installs the websockets/msgpack deps):
uv pip install -e plugins/lerobot_policy_vllm

# Verify both registrations appear:
.venv/bin/lerobot-eval --help | grep -E "policy.type|env.type" | head -2
# → --policy.type   {... vllm ...}
# → --env.type      {... maniskill libero ...}
```

## 1. Run a full eval (LIBERO, real GR00T server on :8000)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/lerobot-eval \
  --policy.path=examples/vllm_policy \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --policy.n_action_steps=10 \
  --output_dir=./outputs/libero_eval
```

## 2. Visualize a single task live (Rerun web viewer)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python \
  examples/vllm_policy/scripts/eval_visualize.py \
  --benchmark libero_object --task-id 0 --open-browser
```

Variants: `--spawn` (native desktop viewer) · `--save /tmp/run.rrd --no-keep-alive`
(offline recording, view with `rerun /tmp/run.rrd`).

## 3. Local mock server (no GPU smoke test)

```bash
.venv/bin/python -m lerobot.policies.vllm.mock_server --host 127.0.0.1 --port 8001
# then re-point the config to the mock:
.venv/bin/python examples/vllm_policy/scripts/generate_policy_dir.py --port 8001
```

## 4. Regenerate this directory

```bash
.venv/bin/python examples/vllm_policy/scripts/generate_policy_dir.py
```

## Contents

```
examples/vllm_policy/
├── README.md                       (this file)
├── config.json                     # {"type": "vllm", host, port, ...}
├── policy_preprocessor.json        # rename + device steps (no normalization)
├── policy_postprocessor.json       # move action to CPU
└── scripts/
    ├── generate_policy_dir.py
    └── eval_visualize.py
```

## Files NOT here

| What | Where |
|---|---|
| `vllm` policy source | `src/lerobot/policies/vllm/` |
| `maniskill` env source | `src/lerobot/envs/maniskill.py` |
| Auto-discovery shim dist | `plugins/lerobot_policy_vllm/` |

## Action / state convention (LIBERO)

Authoritative source: `Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py` +
`examples/LIBERO/modality.json`.

- Embodiment: `libero_sim` (= `LIBERO_PANDA`). Requires the post-trained checkpoint
  `nvidia/GR00T-N1.7-LIBERO/<suite>` on the server (e.g. `.../libero_object`).
- Action: native LIBERO 7-D `[x,y,z,roll,pitch,yaw,gripper]` — fed as **delta** to
  `env.step` (relative control); gripper is `normalize [0,1]→[-1,1]` → `sign` → `invert`.
- Images sent as an ordered list `[agentview, eye_in_hand]` (server feeds HF image
  processor; dict rejected).
- Serialization: vendored OpenPI msgpack-numpy format (`__ndarray__` envelope). The
  standalone `msgpack-numpy` PyPI package is NOT wire-compatible.
- Legacy DROID `eef_9d` mode is still available via `--policy.action_space=eef_9d`.

Tunables: `--policy.host/port`, `--policy.embodiment`, `--policy.action_space`,
`--policy.decode_subtract_state`, `--policy.libero_gripper_binarize`,
`--policy.libero_gripper_invert`, `--policy.action_scale`, `--policy.request_seed`.
