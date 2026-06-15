#!/usr/bin/env python
"""Closed-loop GR00T policy control of a real SO-100/SO-101 over a remote server.

Drives the real robot (lerobot hardware) and queries a running GR00T inference
server (gr00t/eval/run_gr00t_server.py) for actions. This is a standalone port of
Isaac-GR00T/gr00t/eval/real_robot/SO100/eval_so100.py with the ZMQ+msgpack client
inlined, so it runs in the lerobot hardware venv WITHOUT installing the gr00t
package.

Server (already running on the GPU box / this machine):
    python gr00t/eval/run_gr00t_server.py \
        --model-path <ckpt> --embodiment-tag NEW_EMBODIMENT --port 6667

Client (this script, in the lerobot hardware venv):
    .venv/bin/python gr00t_eval.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5AE60583301 \
        --robot.id=my_awesome_follower_arm \
        --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, rotation: 180}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
        --policy_host=127.0.0.1 --policy_port=6667 \
        --lang_instruction="<the task the model was trained on>"

SAFETY: this MOVES the real arm autonomously from model output. Keep a hand on the
power switch; Ctrl-C stops the loop.
"""

import io
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pprint import pformat
from typing import Any
from urllib.parse import unquote

import draccus
import msgpack
import numpy as np
import zmq

# Importing the robot configs registers them for draccus CLI parsing.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so_follower,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


# =============================================================================
# Robust connect — the SO-100 motor bus over flaky USB sometimes drops the first
# handshake ("no status packet"). A broadcast ping wakes the bus; retry a few
# times before giving up.
# =============================================================================
def _wake_bus(port: str) -> None:
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        motors = {f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in range(1, 7)}
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect(handshake=False)
        found = bus.broadcast_ping()
        bus.disconnect()
        logger.info(f"Woke motor bus on {port}: {found}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"bus wake-ping failed ({e})")


def connect_with_retry(robot: Robot, port: str | None, attempts: int = 6) -> None:
    # Besides the flaky motor bus (ConnectionError), the cameras fail transiently at
    # startup: the 1080p top cam needs ~2-3s after being released (by a previous run
    # or a failed attempt) before it accepts a new 640x480 mode set, otherwise
    # connect raises RuntimeError (wrong negotiated size) or TimeoutError (no frame).
    for i in range(1, attempts + 1):
        try:
            robot.connect()
            return
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.warning(f"robot.connect attempt {i}/{attempts} failed ({e}); retrying")
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass
            if isinstance(e, ConnectionError) and port:
                _wake_bus(port)
            time.sleep(3.0)  # let the cameras settle before re-negotiating modes
    robot.connect()  # final attempt — let it raise if it still fails


def revive_dead_cameras(robot: Robot) -> None:
    """Reconnect any camera whose background read thread has died.

    OpenCVCamera's read loop exits permanently after a few consecutive bad frames
    (e.g. the top cam renegotiating its mode under USB bandwidth pressure), after
    which every async_read times out forever. Probe each camera and rebuild the
    dead ones in place.
    """
    for name, cam in getattr(robot, "cameras", {}).items():
        try:
            cam.async_read(timeout_ms=500)
            continue  # alive
        except Exception:  # noqa: BLE001
            pass
        logger.warning(f"camera {name} unresponsive; reconnecting it")
        try:
            cam.disconnect()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3.0)  # settle time before the mode set, same as at startup
        try:
            cam.connect()
            logger.info(f"camera {name} reconnected")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"camera {name} reconnect failed ({e}); will retry on next failure")


def stiffen_arm(robot: Robot, p_gain: int) -> None:
    """Raise the position-loop P gain after connect.

    so_follower.configure() lowers P_Coefficient to 16 (from the servo default 32)
    to reduce teleop shakiness, but that makes the arm too soft to hold/lift its own
    weight under a policy's position targets (it sags below target). Writing a higher
    P stiffens position control so it can actually lift. Does NOT modify the robot
    class — just writes the (RAM) register on the live bus.
    """
    bus = getattr(robot, "bus", None) or getattr(getattr(robot, "_inner", None), "bus", None)
    if bus is None:
        logger.warning("could not access motor bus to set P gain")
        return
    for motor in bus.motors:
        try:
            bus.write("P_Coefficient", motor, int(p_gain))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to set P gain on {motor}: {e}")
    logger.info(f"Set position-loop P_Coefficient={p_gain} on all motors (was 16).")


# =============================================================================
# Live camera view — tiny MJPEG web server (same pattern as the web_so100
# teleop): a daemon encoder thread JPEG-encodes the cameras' latest frames at
# ~15 fps, and each /camera/<name> request streams them as multipart MJPEG.
# Read-only with respect to the robot; the eval control loop is untouched.
# =============================================================================
_VIEW_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>GR00T Eval — Live Cameras</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; padding: 16px; font-family: -apple-system, system-ui, sans-serif; }
  h1 { font-size: 16px; margin: 0 0 4px; opacity: 0.7; font-weight: 500; }
  .task { font-size: 13px; opacity: 0.6; margin: 0 0 12px; font-family: ui-monospace, monospace; }
  .cams { display: flex; gap: 12px; flex-wrap: wrap; }
  .cam { display: flex; flex-direction: column; gap: 4px; }
  .cam img { width: 480px; max-width: 95vw; height: auto; background: #000;
             border-radius: 10px; display: block; border: 1px solid rgba(127,127,127,0.3); }
  .camlabel { font-size: 12px; opacity: 0.6; font-family: ui-monospace, monospace; }
</style>
</head>
<body>
<h1>GR00T Eval — Live Cameras</h1>
<div class="task" id="task"></div>
<div class="cams" id="cams"></div>
<script>
async function init() {
  let info = null;
  try {
    const r = await fetch('/info');
    if (r.ok) info = await r.json();
  } catch (e) { /* server not ready */ }
  if (!info || !info.cameras.length) { setTimeout(init, 700); return; }
  document.getElementById('task').textContent = info.task || '';
  const box = document.getElementById('cams');
  box.innerHTML = '';
  for (const n of info.cameras) {
    const wrap = document.createElement('div'); wrap.className = 'cam';
    const lab = document.createElement('div'); lab.className = 'camlabel'; lab.textContent = n;
    const img = document.createElement('img'); img.alt = n;
    img.src = '/camera/' + encodeURIComponent(n);
    wrap.appendChild(lab); wrap.appendChild(img); box.appendChild(wrap);
  }
}
init();
</script>
</body>
</html>
"""


class CameraStreamServer:
    """Serve the robot's camera feeds as MJPEG streams on a small web page."""

    def __init__(self, host: str, port: int, cameras: dict[str, Any], task: str = ""):
        self._host, self._port = host, port
        self._cameras = dict(cameras)
        self._task = task
        self._frames: dict[str, bytes] = {}
        self._frame_lock = threading.Lock()
        self.closing = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._encoder_thread: threading.Thread | None = None

    # -- frame store ----------------------------------------------------------
    def _set_frame(self, name: str, jpeg: bytes) -> None:
        with self._frame_lock:
            self._frames[name] = jpeg

    def _get_frame(self, name: str) -> bytes | None:
        with self._frame_lock:
            return self._frames.get(name)

    def _camera_names(self) -> list[str]:
        with self._frame_lock:
            return sorted(self._frames)

    # -- encoder --------------------------------------------------------------
    def _encode_loop(self) -> None:
        import cv2

        while not self.closing.is_set():
            for name, cam in self._cameras.items():
                lock = getattr(cam, "frame_lock", None)
                frame = None
                if lock is not None:
                    with lock:
                        frame = getattr(cam, "latest_frame", None)
                if frame is None:
                    continue
                # Observations are RGB; cv2 encodes assuming BGR, so convert first.
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    self._set_frame(name, buf.tobytes())
            self.closing.wait(1.0 / 15.0)

    # -- http -----------------------------------------------------------------
    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_bytes(self, data: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send_bytes(_VIEW_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif self.path == "/info":
                    payload = {"cameras": outer._camera_names(), "task": outer._task}
                    self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json")
                elif self.path.startswith("/camera/"):
                    self._stream_camera(unquote(self.path[len("/camera/") :]))
                else:
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()

            def _stream_camera(self, name: str) -> None:
                if name not in outer._camera_names():
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while not outer.closing.is_set():
                        frame = outer._get_frame(name)
                        if frame is None or frame is last:
                            time.sleep(0.03)
                            continue
                        last = frame
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.04)  # cap browser refresh at ~25 fps
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        return Handler

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        self.closing.clear()
        self._server = ThreadingHTTPServer((self._host, self._port), self._make_handler())
        self._http_thread = threading.Thread(
            target=self._server.serve_forever, name="groot-eval-http", daemon=True
        )
        self._http_thread.start()
        self._encoder_thread = threading.Thread(
            target=self._encode_loop, name="groot-eval-encoder", daemon=True
        )
        self._encoder_thread.start()
        display_host = "localhost" if self._host in ("0.0.0.0", "127.0.0.1") else self._host
        logger.info("Live camera view at http://%s:%d/", display_host, self._port)

    def stop(self) -> None:
        self.closing.set()
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2.0)
            self._encoder_thread = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
            self._http_thread = None


# Joint order used for state/action throughout (matches So100Adapter).
ROBOT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def read_pose(robot: Robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {k: float(obs[k]) for k in ROBOT_STATE_KEYS}


def save_home(robot: Robot, path: str) -> dict[str, float]:
    """Record the arm's current pose as the home pose and persist it to ``path``."""
    pose = read_pose(robot)
    with open(path, "w") as f:
        json.dump(pose, f, indent=2)
    logger.info("Saved current pose as home -> %s: %s", path, {k: round(v, 1) for k, v in pose.items()})
    return pose


def load_home(path: str) -> dict[str, float] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        pose = json.load(f)
    return {k: float(pose[k]) for k in ROBOT_STATE_KEYS if k in pose}


def go_to_pose(robot: Robot, target: dict[str, float], duration_s: float = 3.0, fps: int = 30) -> None:
    """Smoothly interpolate from the current pose to ``target`` (per-joint deg)."""
    start = read_pose(robot)
    steps = max(1, int(duration_s * fps))
    logger.info("Moving to home over %.1fs -> %s", duration_s, {k: round(v, 1) for k, v in target.items()})
    for i in range(1, steps + 1):
        a = i / steps
        cmd = {k: (1.0 - a) * start[k] + a * target[k] for k in ROBOT_STATE_KEYS if k in target}
        try:
            robot.send_action(cmd)
        except ConnectionError as e:
            logger.warning(f"homing send_action hiccup ({e})")
        time.sleep(1.0 / fps)
    logger.info("Reached home pose.")


# =============================================================================
# Minimal GR00T policy client (ZMQ REQ + msgpack) — matches gr00t PolicyServer.
# =============================================================================
def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class GrootClient:
    """Tiny stand-in for gr00t.policy.server_client.PolicyClient (get_action only)."""

    def __init__(self, host: str, port: int, timeout_ms: int = 15000):
        self._ctx = zmq.Context()
        self._host, self._port, self._timeout_ms = host, port, timeout_ms
        self._connect()

    def _connect(self):
        self.socket = self._ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self.socket.connect(f"tcp://{self._host}:{self._port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.socket.send(msgpack.packb(request, default=_encode))
            message = self.socket.recv()
        except zmq.error.Again:
            self._connect()  # REQ socket stuck after timeout; rebuild it
            raise
        response = msgpack.unpackb(message, object_hook=_decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._connect()
            return False

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        resp = self._call("get_action", {"observation": observation, "options": None})
        return tuple(resp)  # (action_chunk, info)


def recursive_add_extra_dim(obs: dict) -> dict:
    """GR00T server expects (batch=1, time=1, ...); call twice to add both dims."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


# =============================================================================
# SO100 adapter (verbatim logic from NVIDIA's eval_so100.py So100Adapter)
# =============================================================================
class So100Adapter:
    def __init__(self, policy_client: GrootClient):
        self.policy = policy_client
        self.robot_state_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        self.camera_keys = ["front", "wrist"]

    def obs_to_policy_inputs(self, obs: dict[str, Any]) -> dict:
        model_obs: dict[str, Any] = {}
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {"single_arm": state[:5], "gripper": state[5:6]}
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: dict, t: int) -> dict[str, float]:
        single_arm = chunk["single_arm"][0][t]  # (5,)
        gripper = chunk["gripper"][0][t]  # (1,)
        full = np.concatenate([single_arm, gripper], axis=0)  # (6,)
        return {name: float(full[i]) for i, name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, _info = self.policy.get_action(model_input)
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) -> T
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Config + main loop
# =============================================================================
@dataclass
class EvalConfig:
    robot: RobotConfig
    policy_host: str = "127.0.0.1"
    policy_port: int = 6667
    action_horizon: int = 8
    lang_instruction: str = "Grab markers and place into pen holder."
    fps: int = 30
    warmup_s: float = 3.0
    # Position-loop P gain written after connect (so_follower lowers it to 16, too
    # soft to lift the arm under policy targets). 32 = servo default; raise if it
    # still sags, lower if it shakes. Set 0 to leave so_follower's value untouched.
    arm_p_gain: int = 32
    # Home pose handling. Record the arm's CURRENT pose as home with --set_home
    # (writes home_pose_file and exits). On a normal run, go_home smoothly moves to
    # that saved pose before the policy takes over, so every run starts identically.
    home_pose_file: str = "so100_home.json"
    set_home: bool = False
    go_home: bool = True
    home_duration_s: float = 3.0
    # Live camera view (MJPEG web page, same as the teleop panel). Open
    # http://<web_host>:<web_port>/ in a browser while the eval runs.
    web: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = 8080


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    connect_with_retry(robot, getattr(cfg.robot, "port", None))

    # --set_home: record the current (manually posed) pose as home, then exit.
    if cfg.set_home:
        save_home(robot, cfg.home_pose_file)
        robot.disconnect()
        logging.info("Home recorded. Re-run without --set_home to evaluate.")
        return

    if cfg.arm_p_gain > 0:
        stiffen_arm(robot, cfg.arm_p_gain)

    # Live camera view in the browser while the policy runs.
    web_server: CameraStreamServer | None = None
    if cfg.web and getattr(robot, "cameras", None):
        web_server = CameraStreamServer(
            cfg.web_host, cfg.web_port, robot.cameras, task=cfg.lang_instruction
        )
        web_server.start()

    # Move to the saved home pose before the policy takes over.
    if cfg.go_home:
        home = load_home(cfg.home_pose_file)
        if home is None:
            logging.warning(
                "No home pose at %s; skipping homing (run with --set_home first). "
                "Policy will start from the current pose.",
                cfg.home_pose_file,
            )
        else:
            logging.warning("Moving to saved home in %.0fs. Clear the workspace. Ctrl-C to abort.", cfg.warmup_s)
            time.sleep(cfg.warmup_s)
            go_to_pose(robot, home, duration_s=cfg.home_duration_s, fps=cfg.fps)

    client = GrootClient(cfg.policy_host, cfg.policy_port)
    logging.info(f"Pinging GR00T server at {cfg.policy_host}:{cfg.policy_port} ...")
    if not client.ping():
        logging.warning("Server did not respond to ping; continuing anyway (it may lack a ping endpoint).")
    else:
        logging.info("Server is up.")
    policy = So100Adapter(client)

    logging.warning(
        "Policy taking over in %.0fs. Instruction: %r. Ctrl-C to stop.",
        cfg.warmup_s,
        cfg.lang_instruction,
    )
    time.sleep(cfg.warmup_s)

    last_obs: dict | None = None
    consec_obs_fails = 0
    try:
        while True:
            try:
                obs = robot.get_observation()
                consec_obs_fails = 0
                last_obs = obs
            except Exception as e:  # noqa: BLE001
                consec_obs_fails += 1
                logging.warning(f"get_observation failed #{consec_obs_fails} ({e})")
                if consec_obs_fails >= 3:
                    revive_dead_cameras(robot)
                    consec_obs_fails = 0
                if last_obs is None:
                    time.sleep(0.5)
                    continue
                obs = last_obs  # reuse last good observation for this cycle
            obs["lang"] = cfg.lang_instruction
            actions = policy.get_action(obs)
            for action_dict in actions[: cfg.action_horizon]:
                tic = time.time()
                robot.send_action(action_dict)
                dt = time.time() - tic
                if dt < 1.0 / cfg.fps:
                    time.sleep(1.0 / cfg.fps - dt)
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        if web_server is not None:
            web_server.stop()
        robot.disconnect()


if __name__ == "__main__":
    eval()
