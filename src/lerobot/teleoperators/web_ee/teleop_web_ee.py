"""Web-button end-effector teleoperator.

Runs a stdlib HTTP server in a daemon thread that serves a small HTML panel
with directional + gripper buttons. Button press/release events update a
shared state dict; ``get_action`` returns the same schema as
``KeyboardEndEffectorTeleop`` so the action drops straight into the
``so100_follower_ee`` robot's IK pipeline.

No extra third-party deps; only ``http.server`` from the stdlib.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .configuration_web_ee import WebEndEffectorTeleopConfig

logger = logging.getLogger(__name__)

# Motors of the SO-100/101 follower, in bus order. The right-side "Joint control"
# panel exposes a ±jog button per motor (``j_<motor>`` axis → ``joint_<motor>`` action).
JOINT_MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
<title>SO-100 EE Teleop</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; touch-action: manipulation; -webkit-user-select: none; user-select: none; }
  body { margin: 0; padding: 16px; font-family: -apple-system, system-ui, sans-serif; }
  h1 { font-size: 16px; margin: 0 0 12px; opacity: 0.7; font-weight: 500; }
  .grid { display: grid; gap: 8px; max-width: 340px; }
  .pad { grid-template-columns: repeat(3, 1fr); aspect-ratio: 3/3; }
  .row { grid-template-columns: 1fr 1fr; }
  button {
    font-size: 22px; padding: 18px 8px; border-radius: 12px;
    border: 1px solid rgba(127,127,127,0.3);
    background: rgba(127,127,127,0.10); cursor: pointer;
    transition: background 0.05s, transform 0.05s;
  }
  button.wide { padding: 14px 8px; font-size: 15px; }
  button.empty { visibility: hidden; }
  button:active, button.held { background: #4c8bf5; color: white; transform: scale(0.97); }
  button.stop { background: rgba(220, 50, 50, 0.10); }
  button.stop:active { background: #dc3232; color: white; }
  button.home { background: rgba(60, 160, 90, 0.12); }
  button.home:active, button.home.flash { background: #3ca05a; color: white; }
  section { margin-top: 14px; }
  .status { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px;
            opacity: 0.6; margin-top: 14px; white-space: pre; }
  .legend { font-size: 11px; opacity: 0.55; margin-top: 6px; max-width: 720px; }
  .cols { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; }
  .col { flex: 0 0 auto; }
  .badge { font-size: 10px; padding: 2px 6px; border-radius: 6px; vertical-align: middle;
           background: rgba(220,50,50,0.18); color: #dc3232; font-weight: 600; }
  .badge.good { background: rgba(60,160,90,0.18); color: #3ca05a; }
</style>
</head>
<body>
<h1>SO-100 Teleop</h1>

<div class="cols">

  <div class="col">
    <h1>EE control · 4-DoF IK <span class="badge">demo · bad case</span></h1>
    <div class="grid pad">
      <button class="empty"></button>
      <button data-axis="dx" data-val="1" title="+X (forward, away from base)">↑</button>
      <button class="empty"></button>
      <button data-axis="dy" data-val="1" title="+Y (arm's left)">←</button>
      <button class="stop" data-halt>STOP</button>
      <button data-axis="dy" data-val="-1" title="−Y (arm's right)">→</button>
      <button class="empty"></button>
      <button data-axis="dx" data-val="-1" title="−X (backward, toward base)">↓</button>
      <button class="empty"></button>
    </div>
    <section class="grid row">
      <button data-axis="dz" data-val="1" class="wide">Z+ (raise)</button>
      <button data-axis="dz" data-val="-1" class="wide">Z− (lower)</button>
    </section>
    <section class="grid row">
      <button data-axis="roll" data-val="1" class="wide" title="wrist_roll + : spin gripper CCW">Roll ↺</button>
      <button data-axis="roll" data-val="-1" class="wide" title="wrist_roll − : spin gripper CW">Roll ↻</button>
    </section>
    <section class="grid row" id="gripper-row">
      <button data-axis="gripper" data-val="2" class="wide">Open ✋</button>
      <button data-axis="gripper" data-val="0" class="wide">Close ✊</button>
    </section>
  </div>

  <div class="col">
    <h1>Joint control · all 6 motors <span class="badge good">recommended</span></h1>
    <section class="grid row">
      <button data-axis="j_shoulder_pan" data-val="-1" class="wide">J1 Pan −</button>
      <button data-axis="j_shoulder_pan" data-val="1" class="wide">J1 Pan +</button>
    </section>
    <section class="grid row">
      <button data-axis="j_shoulder_lift" data-val="-1" class="wide">J2 Lift −</button>
      <button data-axis="j_shoulder_lift" data-val="1" class="wide">J2 Lift +</button>
    </section>
    <section class="grid row">
      <button data-axis="j_elbow_flex" data-val="-1" class="wide">J3 Elbow −</button>
      <button data-axis="j_elbow_flex" data-val="1" class="wide">J3 Elbow +</button>
    </section>
    <section class="grid row">
      <button data-axis="j_wrist_flex" data-val="-1" class="wide">J4 W.Flex −</button>
      <button data-axis="j_wrist_flex" data-val="1" class="wide">J4 W.Flex +</button>
    </section>
    <section class="grid row">
      <button data-axis="j_wrist_roll" data-val="-1" class="wide">J5 W.Roll −</button>
      <button data-axis="j_wrist_roll" data-val="1" class="wide">J5 W.Roll +</button>
    </section>
    <section class="grid row">
      <button data-axis="j_gripper" data-val="-1" class="wide">J6 Grip −</button>
      <button data-axis="j_gripper" data-val="1" class="wide">J6 Grip +</button>
    </section>
  </div>

</div>

<section class="grid">
  <button class="home wide" id="home-btn">HOME (reset all motors to neutral)</button>
</section>

<div class="status" id="status">dx=0 dy=0 dz=0 roll=0 gripper=1</div>
<div class="legend"><b>Left — EE control (4-DoF IK), kept as a demo of why EE control is a poor fit for this 5-DoF arm.</b>
↑/↓ = ±X, ←/→ = ±Y, Z± = up/down; the IK auto-holds the gripper pitch level (no droop); Roll = wrist_roll.
<b>Right — direct joint jog of all 6 motors (recommended).</b> Each ± nudges one motor; bypasses IK entirely.
Hold any button to keep moving. STOP releases all axes. HOME snaps every motor to its calibrated middle.</div>

<script>
const status = document.getElementById('status');
let state = { dx:0, dy:0, dz:0, roll:0, gripper:1 };

function renderStatus() {
  status.textContent = `dx=${state.dx} dy=${state.dy} dz=${state.dz} roll=${state.roll} gripper=${state.gripper}`;
}

async function send(path, body) {
  try {
    const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                                  body: JSON.stringify(body || {})});
    if (r.ok) { state = await r.json(); renderStatus(); }
  } catch (e) { console.error(e); }
}

function press(btn) {
  if (btn.dataset.halt !== undefined) { btn.classList.add('held'); send('/halt'); return; }
  btn.classList.add('held');
  send('/press', { axis: btn.dataset.axis, val: Number(btn.dataset.val) });
}
function release(btn) {
  btn.classList.remove('held');
  if (btn.dataset.halt !== undefined) return;
  send('/release', { axis: btn.dataset.axis });
}

for (const btn of document.querySelectorAll('button[data-axis], button[data-halt]')) {
  btn.addEventListener('mousedown', e => { e.preventDefault(); press(btn); });
  btn.addEventListener('mouseup',   e => { e.preventDefault(); release(btn); });
  btn.addEventListener('mouseleave', e => { if (btn.classList.contains('held')) release(btn); });
  btn.addEventListener('touchstart', e => { e.preventDefault(); press(btn); }, {passive:false});
  btn.addEventListener('touchend',   e => { e.preventDefault(); release(btn); }, {passive:false});
  btn.addEventListener('touchcancel',e => { e.preventDefault(); release(btn); }, {passive:false});
}

// Safety net: if any mouseup happens off-button, release everything.
window.addEventListener('mouseup', () => {
  for (const b of document.querySelectorAll('button.held')) b.classList.remove('held');
});

window.addEventListener('blur', () => send('/halt'));

const homeBtn = document.getElementById('home-btn');
homeBtn.addEventListener('click', async () => {
  homeBtn.classList.add('flash');
  await send('/home');
  setTimeout(() => homeBtn.classList.remove('flash'), 400);
});

renderStatus();
</script>
</body>
</html>
"""


class _State:
    """Thread-safe shared state for the button presses."""

    def __init__(self, use_gripper: bool):
        self._lock = threading.Lock()
        self._use_gripper = use_gripper
        # EE-control panel (left): dx/dy/dz and roll are -1/0/+1 ; gripper is 0/1/2
        # (close/stay/open) matching keyboard_ee. Joint panel (right): j_<motor> are
        # -1/0/+1 per-motor jog directions that bypass IK on the robot side.
        self._state: dict[str, int] = {
            "dx": 0,
            "dy": 0,
            "dz": 0,
            "roll": 0,
            "gripper": 1,
        }
        self._state.update({f"j_{m}": 0 for m in JOINT_MOTORS})
        # One-shot flag — set by /home, cleared the next time get_action() snapshots it.
        self._home_request: bool = False

    def request_home(self) -> dict[str, int]:
        with self._lock:
            self._home_request = True
            return {**self._state, "home": True}

    def consume_home(self) -> bool:
        with self._lock:
            requested = self._home_request
            self._home_request = False
            return requested

    def press(self, axis: str, val: int) -> dict[str, int]:
        if axis not in self._state:
            raise ValueError(f"unknown axis: {axis}")
        if axis == "gripper":
            if val not in (0, 1, 2):
                raise ValueError(f"gripper val must be 0/1/2, got {val}")
        else:
            if val not in (-1, 0, 1):
                raise ValueError(f"delta val must be -1/0/+1, got {val}")
        with self._lock:
            self._state[axis] = val
            return dict(self._state)

    def release(self, axis: str) -> dict[str, int]:
        if axis not in self._state:
            raise ValueError(f"unknown axis: {axis}")
        with self._lock:
            self._state[axis] = 1 if axis == "gripper" else 0
            return dict(self._state)

    def halt(self) -> dict[str, int]:
        with self._lock:
            self._state = {"dx": 0, "dy": 0, "dz": 0, "roll": 0, "gripper": 1}
            self._state.update({f"j_{m}": 0 for m in JOINT_MOTORS})
            return dict(self._state)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._state)


def _make_handler(state: _State, html: str):
    class Handler(BaseHTTPRequestHandler):
        # Quieter logs — only warnings/errors.
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8") or "{}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state":
                self._send_json(state.snapshot())
            else:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()

        def do_POST(self):
            try:
                body = self._read_json()
                if self.path == "/press":
                    payload = state.press(str(body["axis"]), int(body["val"]))
                elif self.path == "/release":
                    payload = state.release(str(body["axis"]))
                elif self.path == "/halt":
                    payload = state.halt()
                elif self.path == "/home":
                    payload = state.request_home()
                else:
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
                    return
                self._send_json(payload)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=400)

    return Handler


class WebEndEffectorTeleop(Teleoperator):
    """Browser-button driven teleop matching the ``keyboard_ee`` action schema."""

    config_class = WebEndEffectorTeleopConfig
    name = "web_ee"

    def __init__(self, config: WebEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self._state = _State(use_gripper=config.use_gripper)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def action_features(self) -> dict:
        keys = {
            "delta_x": 0,
            "delta_y": 1,
            "delta_z": 2,
            "delta_roll": 3,
        }
        if self.config.use_gripper:
            keys["gripper"] = len(keys)
        for motor in JOINT_MOTORS:
            keys[f"joint_{motor}"] = len(keys)
        return {
            "dtype": "float32",
            "shape": (len(keys),),
            "names": keys,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        handler_cls = _make_handler(self._state, _INDEX_HTML)
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="web-ee-http",
            daemon=True,
        )
        self._thread.start()
        # Give the server a fraction of a second to bind before logging.
        time.sleep(0.05)
        host = self.config.host
        display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
        logger.info(
            "%s WebEndEffectorTeleop serving control panel at http://%s:%d/",
            self.id,
            display_host,
            self.config.port,
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self._state.consume_home():
            # One-shot: tell the robot to bypass IK and snap to its home pose.
            return {"home": True}
        snap = self._state.snapshot()
        action: dict[str, Any] = {
            "delta_x": float(snap["dx"]),
            "delta_y": float(snap["dy"]),
            "delta_z": float(snap["dz"]),
            "delta_roll": float(snap["roll"]),
        }
        if self.config.use_gripper:
            action["gripper"] = float(snap["gripper"])
        # Direct per-motor jog (right panel) — bypasses IK on the robot side.
        for motor in JOINT_MOTORS:
            action[f"joint_{motor}"] = float(snap[f"j_{motor}"])
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        assert self._server is not None
        # shutdown() blocks until serve_forever exits; safe to call from another thread.
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
        logger.info("%s WebEndEffectorTeleop disconnected.", self.id)
