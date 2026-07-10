"""Joint-only web teleoperator for the plain SO-100/101 follower.

Runs a stdlib HTTP server in a daemon thread that serves a small HTML panel with
a +/- jog button per motor. There is NO inverse kinematics: the teleop keeps an
absolute target position per motor (seeded from the robot's current pose via
``attach_robot``) and each held button ramps that target at ``jog_speed_deg_s``.
``get_action`` returns ``{"<motor>.pos": target_deg}`` which drives the original
``so_follower`` directly — the robot itself is never modified or subclassed.

Robot cameras are streamed live (MJPEG) at the top of the same panel, encoded off
the control thread and read straight from the cameras so the view stays smooth
even when the control loop hitches.

No extra third-party deps; only ``http.server`` from the stdlib (plus cv2/numpy
for the camera stream, which lerobot already depends on).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .configuration_web_so100 import WebSO100TeleopConfig

logger = logging.getLogger(__name__)

# Motors of the SO-100/101 follower, in bus order.
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
<title>SO-100 Joint Teleop</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; touch-action: manipulation; -webkit-user-select: none; user-select: none; }
  body { margin: 0; padding: 16px; font-family: -apple-system, system-ui, sans-serif; }
  h1 { font-size: 16px; margin: 0 0 12px; opacity: 0.7; font-weight: 500; }
  .grid { display: grid; gap: 8px; max-width: 360px; }
  .row { grid-template-columns: 1fr 1fr; }
  button {
    font-size: 16px; padding: 16px 8px; border-radius: 12px;
    border: 1px solid rgba(127,127,127,0.3);
    background: rgba(127,127,127,0.10); cursor: pointer;
    transition: background 0.05s, transform 0.05s;
  }
  button.empty { visibility: hidden; }
  button:active, button.held { background: #4c8bf5; color: white; transform: scale(0.97); }
  button.stop { background: rgba(220, 50, 50, 0.10); }
  button.stop:active, button.stop.held { background: #dc3232; color: white; }
  button.home { background: rgba(60, 160, 90, 0.12); }
  button.home:active, button.home.flash { background: #3ca05a; color: white; }
  section { margin-top: 10px; }
  .status { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px;
            opacity: 0.6; margin-top: 14px; white-space: pre; }
  .legend { font-size: 11px; opacity: 0.55; margin-top: 8px; max-width: 720px; }
  .cams { display: flex; gap: 12px; flex-wrap: wrap; margin: 0 0 16px; }
  .cam { display: flex; flex-direction: column; gap: 4px; }
  .cam img { width: 360px; max-width: 90vw; height: auto;
             background: #000; border-radius: 10px; display: block;
             border: 1px solid rgba(127,127,127,0.3); }
  .camlabel { font-size: 12px; opacity: 0.6; font-family: ui-monospace, monospace; }
</style>
</head>
<body>
<h1>SO-100 Joint Teleop (no IK)</h1>

<div class="cams" id="cams"></div>

<div class="col">
  <h1>Joint control · all 6 motors</h1>
  <section class="grid row">
    <button data-axis="j_shoulder_pan" data-val="-1">J1 Pan −</button>
    <button data-axis="j_shoulder_pan" data-val="1">J1 Pan +</button>
  </section>
  <section class="grid row">
    <button data-axis="j_shoulder_lift" data-val="-1">J2 Lift −</button>
    <button data-axis="j_shoulder_lift" data-val="1">J2 Lift +</button>
  </section>
  <section class="grid row">
    <button data-axis="j_elbow_flex" data-val="-1">J3 Elbow −</button>
    <button data-axis="j_elbow_flex" data-val="1">J3 Elbow +</button>
  </section>
  <section class="grid row">
    <button data-axis="j_wrist_flex" data-val="-1">J4 W.Flex −</button>
    <button data-axis="j_wrist_flex" data-val="1">J4 W.Flex +</button>
  </section>
  <section class="grid row">
    <button data-axis="j_wrist_roll" data-val="-1">J5 W.Roll −</button>
    <button data-axis="j_wrist_roll" data-val="1">J5 W.Roll +</button>
  </section>
  <section class="grid row">
    <button data-axis="j_gripper" data-val="-1">J6 Grip −</button>
    <button data-axis="j_gripper" data-val="1">J6 Grip +</button>
  </section>
</div>

<section class="grid">
  <button class="stop" id="stop-btn">STOP (release all)</button>
</section>
<section class="grid">
  <button class="home" id="home-btn">HOME (all motors to center 0°)</button>
</section>

<div class="status" id="status">all axes 0</div>
<div class="legend">Hold a button to move that motor; the target ramps while held and holds when
released. STOP releases all axes. HOME drives every motor to its calibrated center (0°).</div>

<script>
const status = document.getElementById('status');
let state = {};

function renderStatus() {
  const on = Object.entries(state).filter(([k,v]) => v).map(([k,v]) => `${k}=${v}`);
  status.textContent = on.length ? on.join(' ') : 'all axes 0';
}

async function send(path, body) {
  try {
    const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                                  body: JSON.stringify(body || {})});
    if (r.ok) { state = await r.json(); renderStatus(); }
  } catch (e) { console.error(e); }
}

function press(btn) {
  btn.classList.add('held');
  send('/press', { axis: btn.dataset.axis, val: Number(btn.dataset.val) });
}
function release(btn) {
  btn.classList.remove('held');
  send('/release', { axis: btn.dataset.axis });
}

for (const btn of document.querySelectorAll('button[data-axis]')) {
  btn.addEventListener('mousedown', e => { e.preventDefault(); press(btn); });
  btn.addEventListener('mouseup',   e => { e.preventDefault(); release(btn); });
  btn.addEventListener('mouseleave', e => { if (btn.classList.contains('held')) release(btn); });
  btn.addEventListener('touchstart', e => { e.preventDefault(); press(btn); }, {passive:false});
  btn.addEventListener('touchend',   e => { e.preventDefault(); release(btn); }, {passive:false});
  btn.addEventListener('touchcancel',e => { e.preventDefault(); release(btn); }, {passive:false});
}

// Safety net: any mouseup off-button releases everything.
window.addEventListener('mouseup', () => {
  for (const b of document.querySelectorAll('button.held')) b.classList.remove('held');
});
window.addEventListener('blur', () => send('/halt'));

document.getElementById('stop-btn').addEventListener('click', () => send('/halt'));
const homeBtn = document.getElementById('home-btn');
homeBtn.addEventListener('click', async () => {
  homeBtn.classList.add('flash');
  await send('/home');
  setTimeout(() => homeBtn.classList.remove('flash'), 400);
});

renderStatus();

// Discover camera feeds and render an auto-updating MJPEG <img> for each.
async function initCameras() {
  let names = [];
  try {
    const r = await fetch('/cameras');
    if (r.ok) names = await r.json();
  } catch (e) { /* server not ready */ }
  if (!names.length) { setTimeout(initCameras, 700); return; }
  const box = document.getElementById('cams');
  box.innerHTML = '';
  for (const n of names) {
    const wrap = document.createElement('div'); wrap.className = 'cam';
    const lab = document.createElement('div'); lab.className = 'camlabel'; lab.textContent = n;
    const img = document.createElement('img'); img.alt = n;
    img.src = '/camera/' + encodeURIComponent(n);
    wrap.appendChild(lab); wrap.appendChild(img); box.appendChild(wrap);
  }
}
initCameras();
</script>
</body>
</html>
"""


class _State:
    """Thread-safe shared state: per-motor jog direction + camera frames."""

    def __init__(self):
        self._lock = threading.Lock()
        # j_<motor> jog direction in {-1, 0, +1}.
        self._jog: dict[str, int] = {f"j_{m}": 0 for m in JOINT_MOTORS}
        # One-shot HOME flag, cleared the next time get_action() reads it.
        self._home_request: bool = False
        # Latest JPEG per camera (name -> bytes), served over MJPEG.
        self._frame_lock = threading.Lock()
        self._frames: dict[str, bytes] = {}
        # Set on disconnect so streaming handlers / encoder exit their loops.
        self.closing = threading.Event()

    def request_home(self) -> dict[str, int]:
        with self._lock:
            self._home_request = True
            return {**self._jog, "home": 1}

    def consume_home(self) -> bool:
        with self._lock:
            requested = self._home_request
            self._home_request = False
            return requested

    def press(self, axis: str, val: int) -> dict[str, int]:
        if axis not in self._jog:
            raise ValueError(f"unknown axis: {axis}")
        if val not in (-1, 0, 1):
            raise ValueError(f"jog val must be -1/0/+1, got {val}")
        with self._lock:
            self._jog[axis] = val
            return dict(self._jog)

    def release(self, axis: str) -> dict[str, int]:
        if axis not in self._jog:
            raise ValueError(f"unknown axis: {axis}")
        with self._lock:
            self._jog[axis] = 0
            return dict(self._jog)

    def halt(self) -> dict[str, int]:
        with self._lock:
            self._jog = {f"j_{m}": 0 for m in JOINT_MOTORS}
            return dict(self._jog)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._jog)

    def set_frame(self, name: str, jpeg: bytes) -> None:
        with self._frame_lock:
            self._frames[name] = jpeg

    def get_frame(self, name: str) -> bytes | None:
        with self._frame_lock:
            return self._frames.get(name)

    def camera_names(self) -> list[str]:
        with self._frame_lock:
            return sorted(self._frames)


def _make_handler(state: _State, html: str):
    class Handler(BaseHTTPRequestHandler):
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

        def _stream_camera(self, name: str) -> None:
            if name not in state.camera_names():
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
                while not state.closing.is_set():
                    frame = state.get_frame(name)
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
            elif self.path == "/cameras":
                data = json.dumps(state.camera_names()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/camera/"):
                self._stream_camera(unquote(self.path[len("/camera/") :]))
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


class WebSO100Teleop(Teleoperator):
    """Browser-button joint-jog teleop that drives the plain so_follower (no IK)."""

    config_class = WebSO100TeleopConfig
    name = "web_so100"

    def __init__(self, config: WebSO100TeleopConfig):
        super().__init__(config)
        self.config = config
        self._state = _State()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # Absolute per-motor target positions (deg). Seeded by attach_robot();
        # None until then so get_action emits no movement before seeding.
        self._targets: dict[str, float] | None = None
        self._targets_lock = threading.Lock()
        self._last_t: float | None = None
        self._warned_unseeded = False
        # Camera streaming (encoded off the control thread; read directly from the
        # robot's cameras so the view stays fresh even if the control loop hitches).
        self._cameras: dict[str, Any] = {}
        self._encoder_thread: threading.Thread | None = None
        self._encode_interval_s = 1.0 / 15.0

    @property
    def action_features(self) -> dict:
        return {f"{motor}.pos": float for motor in JOINT_MOTORS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._state.closing.clear()
        handler_cls = _make_handler(self._state, _INDEX_HTML)
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever, name="web-so100-http", daemon=True)
        self._thread.start()
        self._encoder_thread = threading.Thread(
            target=self._encode_loop, name="web-so100-encoder", daemon=True
        )
        self._encoder_thread.start()
        time.sleep(0.05)
        host = self.config.host
        display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
        logger.info(
            "%s WebSO100Teleop serving control panel at http://%s:%d/", self.id, display_host, self.config.port
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def attach_robot(self, robot) -> None:
        """Seed target positions from the robot's current pose and grab its cameras.

        Called once after the robot connects. The teleop only *reads* from the
        robot (observation for seeding, cameras for the live view); it never
        modifies or subclasses it.
        """
        # Seed absolute targets from the current joint positions so the arm holds
        # still until the user jogs (rather than snapping to 0 on the first tick).
        try:
            obs = robot.get_observation()
            targets = {m: float(obs[f"{m}.pos"]) for m in JOINT_MOTORS if f"{m}.pos" in obs}
            if len(targets) == len(JOINT_MOTORS):
                with self._targets_lock:
                    self._targets = targets
                logger.info("%s seeded joint targets from robot pose.", self.id)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s could not seed joint targets (%s); will hold until seeded.", self.id, e)
        # Cameras for the live view.
        cams = getattr(robot, "cameras", None) or getattr(getattr(robot, "_inner", None), "cameras", None)
        self._cameras = dict(cams) if cams else {}

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.perf_counter()
        dt = (now - self._last_t) if self._last_t is not None else 0.0
        self._last_t = now

        with self._targets_lock:
            if self._targets is None:
                if not self._warned_unseeded:
                    self._warned_unseeded = True
                    logger.warning("%s get_action before targets seeded; sending no motion.", self.id)
                return {}
            if self._state.consume_home():
                for m in JOINT_MOTORS:
                    self._targets[m] = 0.0
            else:
                jog = self._state.snapshot()
                step = self.config.jog_speed_deg_s * dt
                for m in JOINT_MOTORS:
                    d = jog.get(f"j_{m}", 0)
                    if d:
                        self._targets[m] += d * step
            return {f"{m}.pos": float(self._targets[m]) for m in JOINT_MOTORS}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback  # camera frames are read directly from the attached cameras

    def _grab_frames(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, cam in self._cameras.items():
            lock = getattr(cam, "frame_lock", None)
            frame = None
            if lock is not None:
                with lock:
                    frame = getattr(cam, "latest_frame", None)
            if frame is not None:
                out[name] = frame
        return out

    def _encode_loop(self) -> None:
        try:
            import cv2
        except ImportError:
            return
        closing = self._state.closing
        while not closing.is_set():
            for name, val in self._grab_frames().items():
                h, w = val.shape[:2]
                frame = cv2.resize(val, (640, int(round(h * 640 / w)))) if w > 640 else val
                # Observations are RGB; cv2 encodes assuming BGR, so convert first.
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    self._state.set_frame(name, buf.tobytes())
            closing.wait(self._encode_interval_s)

    @check_if_not_connected
    def disconnect(self) -> None:
        assert self._server is not None
        self._state.closing.set()
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2.0)
            self._encoder_thread = None
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
        logger.info("%s WebSO100Teleop disconnected.", self.id)
