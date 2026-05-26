"""VllmPolicy: a LeRobot policy that delegates action inference to a remote vLLM
OpenPI WebSocket server (e.g. vllm-omni GR00T-N1.7).

It does not hold model weights. At each step it encodes the (LIBERO) observation into
an OpenPI request, sends it to the remote server, and decodes the returned *absolute*
action trajectory into LIBERO's 7-D action space, buffering ``n_action_steps`` actions.
"""

from __future__ import annotations

import builtins
import logging
import uuid
from collections import deque
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .client import OpenPIClient
from .configuration_vllm import VllmConfig
from .encoding import (
    axisangle_to_matrix,
    chw01_to_hwc_uint8,
    eef_9d_from_pos_axisangle,
    gripper_position_to_action,
    gripper_qpos_to_position,
    matrix_to_axisangle,
    rot6d_to_matrix,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound="VllmPolicy")


def _resize_hwc(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:  # pragma: no cover - fallback nearest-neighbour
        ys = (np.linspace(0, img.shape[0] - 1, height)).astype(int)
        xs = (np.linspace(0, img.shape[1] - 1, width)).astype(int)
        return img[ys][:, xs]


class VllmPolicy(PreTrainedPolicy):
    """Remote-inference policy proxying to a vLLM OpenPI server."""

    name = "vllm"
    config_class = VllmConfig

    def __init__(self, config: VllmConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        # Dummy buffer so the module has a device and `.to(...)`/`.parameters()` behave.
        self.register_buffer("_device_marker", torch.zeros(1), persistent=False)
        self._client = OpenPIClient(
            url=config.ws_url,
            connect_timeout_s=config.connect_timeout_s,
            max_msg_bytes=config.max_msg_bytes,
        )
        self._server_image_hw: tuple[int, int] | None = None
        self.reset()

    # --- lifecycle ---
    def reset(self):
        """Clear the action buffer and start a fresh server session."""
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)
        self._session_id = str(uuid.uuid4())

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: VllmConfig | None = None,
        **kwargs,
    ) -> T:
        """Build the policy from config only (no weights to load for a remote policy)."""
        if config is None:
            config = VllmConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        policy = cls(config)
        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        raise NotImplementedError("VllmPolicy is inference-only (remote server); training is unsupported.")

    # --- inference ---
    @property
    def _device(self) -> torch.device:
        return self._device_marker.device

    def _image_hw(self) -> tuple[int, int]:
        if self._server_image_hw is not None:
            return self._server_image_hw
        return (self.config.image_height, self.config.image_width)

    def _maybe_handshake(self) -> None:
        """Fetch server image resolution once (best-effort)."""
        if self._server_image_hw is not None:
            return
        try:
            meta = self._client.handshake()
            res = meta.get("image_resolution")
            if res and len(res) == 2:
                self._server_image_hw = (int(res[0]), int(res[1]))
        except Exception as exc:  # pragma: no cover - non-fatal
            logger.debug("Server handshake for image resolution failed: %s", exc)
            self._server_image_hw = (self.config.image_height, self.config.image_width)

    def _build_request(self, batch: dict[str, Tensor], i: int) -> dict[str, Any]:
        cfg = self.config
        state = batch[cfg.state_obs_key][i].detach().cpu().numpy().astype(np.float64).reshape(-1)

        if cfg.action_space == "libero_7d":
            # LeRobot LiberoProcessorStep state = [eef_pos(3), eef_axisangle(3), gripper_qpos(2)],
            # which is exactly the LIBERO state [x,y,z,roll,pitch,yaw,gripper(2)].
            raw_state = {
                "x": [float(state[0])],
                "y": [float(state[1])],
                "z": [float(state[2])],
                "roll": [float(state[3])],
                "pitch": [float(state[4])],
                "yaw": [float(state[5])],
                "gripper": [float(state[6]), float(state[7])],
            }
        else:  # legacy DROID eef_9d
            eef_9d = eef_9d_from_pos_axisangle(state[0:3], state[3:6])
            gripper_position = gripper_qpos_to_position(
                state[6:8], cfg.gripper_qpos_open, cfg.gripper_qpos_closed
            )
            raw_state = {
                "eef_9d": [float(x) for x in eef_9d],
                "gripper_position": [float(gripper_position)],
            }
            if cfg.send_joint_position:
                raw_state["joint_position"] = [0.0] * cfg.joint_dim

        h, w = self._image_hw()
        ext = chw01_to_hwc_uint8(batch[cfg.external_camera_obs_key][i].detach().cpu().numpy(), cfg.flip_images_180)
        ordered: list[tuple[str, np.ndarray]] = [("external_0", _resize_hwc(ext, h, w))]
        if cfg.send_wrist_camera and cfg.wrist_camera_obs_key in batch:
            wrist = chw01_to_hwc_uint8(
                batch[cfg.wrist_camera_obs_key][i].detach().cpu().numpy(), cfg.flip_images_180
            )
            ordered.append(("wrist", _resize_hwc(wrist, h, w)))
        # The deployed server requires a list (or single image); a dict raises in the HF
        # image processor. `images_as_list` keeps a dict path available for other servers.
        images: Any = [im for _, im in ordered] if cfg.images_as_list else dict(ordered)

        prompt = cfg.prompt_override
        if prompt is None:
            task = batch.get("task")
            prompt = task[i] if isinstance(task, (list, tuple)) and i < len(task) else (task or "")

        obs: dict[str, Any] = {
            "session_id": self._session_id,
            "embodiment": cfg.embodiment,
            "modality_config": cfg.modality_config,
            "prompt": prompt,
            "state": raw_state,
            "images": images,
        }
        if cfg.request_seed is not None:
            obs["seed"] = int(cfg.request_seed)
        return obs

    def _decode_libero_7d(self, actions: dict[str, np.ndarray], state_i: np.ndarray) -> np.ndarray:
        """LIBERO: build the native 7-D OSC_POSE action [x,y,z,roll,pitch,yaw,gripper].

        Mirrors Isaac-GR00T's `gr00t/eval/sim/LIBERO/libero_env.py`:
          - the model output is a per-step DELTA fed straight to env.step (relative control);
          - the gripper is normalized [0,1]->[-1,1], optionally sign-binarized, then inverted.
        The vllm-omni server returns ABSOLUTE values (current state + delta), so when
        `decode_subtract_state` is set we subtract the current state to recover the delta.
        """
        cfg = self.config
        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        present = [k for k in pose_keys if k in actions]
        if not present:
            raise RuntimeError(f"Server response missing LIBERO action keys; got {sorted(actions.keys())}")
        horizon = min(int(np.asarray(actions[k]).shape[0]) for k in present)
        n = min(cfg.n_action_steps, horizon)
        out = np.zeros((n, 7), dtype=np.float32)

        for idx, key in enumerate(pose_keys):
            col = np.asarray(actions[key], dtype=np.float32).reshape(horizon, -1)[:n, 0]
            if cfg.decode_subtract_state:
                col = col - np.float32(state_i[idx])
            out[:, idx] = col * cfg.action_scale

        # gripper: recover raw model output, then normalize/binarize/invert.
        g = np.asarray(actions["gripper"], dtype=np.float32).reshape(horizon, -1)[:n, 0]
        if cfg.decode_subtract_state:
            g = g - np.float32(state_i[6])  # undo the server's +raw_state["gripper"][0]
        g = 2.0 * g - 1.0  # normalize [0,1] -> [-1,1]
        if cfg.libero_gripper_binarize:
            g = np.sign(g)
        if cfg.libero_gripper_invert:
            g = -g
        out[:, 6] = g

        if cfg.clip_action:
            np.clip(out, -1.0, 1.0, out=out)
        return out

    def _decode_actions(self, actions: dict[str, np.ndarray], state_i: np.ndarray) -> np.ndarray:
        """Server action dict -> LIBERO action chunk (n, 7)."""
        cfg = self.config
        if cfg.action_space == "libero_7d":
            return self._decode_libero_7d(actions, state_i)
        eef_traj = actions.get("eef_9d")
        if eef_traj is None:
            raise RuntimeError(f"Server response missing 'eef_9d'; got keys {sorted(actions.keys())}")
        eef_traj = np.asarray(eef_traj, dtype=np.float64)
        grip_traj = actions.get("gripper_position")
        grip_traj = np.asarray(grip_traj, dtype=np.float64).reshape(-1) if grip_traj is not None else None

        horizon = eef_traj.shape[0]
        n = min(cfg.n_action_steps, horizon)
        out = np.zeros((n, 7), dtype=np.float32)

        prev_xyz = np.asarray(state_i[0:3], dtype=np.float64)
        prev_R = axisangle_to_matrix(state_i[3:6])

        for j in range(n):
            tgt_xyz = eef_traj[j, 0:3]
            tgt_R = rot6d_to_matrix(eef_traj[j, 3:9])
            if cfg.control_mode == "relative":
                out[j, 0:3] = (tgt_xyz - prev_xyz) * cfg.action_scale
                out[j, 3:6] = matrix_to_axisangle(tgt_R @ prev_R.T) * cfg.action_scale
                prev_xyz, prev_R = tgt_xyz, tgt_R
            else:  # absolute
                out[j, 0:3] = tgt_xyz
                out[j, 3:6] = matrix_to_axisangle(tgt_R)
            g = float(grip_traj[j]) if grip_traj is not None and j < grip_traj.shape[0] else 0.0
            out[j, 6] = gripper_position_to_action(g, cfg.gripper_action_open, cfg.gripper_action_close)

        if cfg.clip_action:
            np.clip(out, -1.0, 1.0, out=out)
        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Query the remote server for each env and return (B, n_action_steps, 7)."""
        self._maybe_handshake()
        cfg = self.config
        batch_size = batch[cfg.state_obs_key].shape[0]

        chunks: list[np.ndarray] = []
        for i in range(batch_size):
            obs = self._build_request(batch, i)
            actions, _meta = self._client.infer(obs)
            state_i = batch[cfg.state_obs_key][i].detach().cpu().numpy().astype(np.float64).reshape(-1)
            chunks.append(self._decode_actions(actions, state_i))

        arr = np.stack(chunks, axis=0)  # (B, n, 7)
        return torch.from_numpy(arr).to(self._device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action (B, 7); refill the buffer from the server when empty."""
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)  # (B, n, 7)
            # queue holds n entries, each (B, 7)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
