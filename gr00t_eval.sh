#!/usr/bin/env bash
# Closed-loop GR00T policy control of the real SO-100/101.
#
# Prereq: the GR00T server is already running and reachable, e.g.:
#   python gr00t/eval/run_gr00t_server.py \
#     --model-path /data/.../checkpoint-19800 \
#     --embodiment-tag NEW_EMBODIMENT --port 6667
#
# Cameras: the MODEL expects names `front` and `wrist`. front = "USB Camera"
# (top-down), wrist = "Innomaker-U20CAM-1080p-S1". macOS shuffles OpenCV indices
# whenever devices come/go (iPhone Continuity Camera, replugs), so resolve them
# by hardware fingerprint at launch instead of hardcoding.
#
# SAFETY: the arm moves autonomously from model output. Keep a hand on the power
# switch; Ctrl-C stops the loop. There is a 3s warmup before motion starts.
IDX=($(.venv/bin/python resolve_cameras.py front wrist)) || exit 1
FRONT_IDX=${IDX[0]}
WRIST_IDX=${IDX[1]}
echo "Resolved camera indices: front=${FRONT_IDX} wrist=${WRIST_IDX}"

# Lock exposure so the image stays constant (auto-exposure "breathes" against the
# white table and blows out highlights). UVC settings reset on replug/reboot, so
# re-apply at every launch. Devices selected by USB vendor:product id (stable).
# front locks white balance too; the Innomaker's manual WB has a green cast, so
# the wrist keeps auto WB. Tuned 2026-06-10: front mean ~170, wrist ~140, 0% clip.
if [ -x ./uvc-util ]; then
    ./uvc-util -V 0x32e6:0x9221 -s auto-exposure-mode=1 -s exposure-time-abs=250 \
               -s auto-white-balance-temp=0 -s white-balance-temp=3300 || true   # front "USB Camera"
    ./uvc-util -V 0x0c45:0x6366 -s auto-exposure-mode=1 -s exposure-time-abs=120 \
               -s auto-white-balance-temp=1 || true                              # wrist Innomaker
fi
.venv/bin/python gr00t_eval.py \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AE60583301 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: ${FRONT_IDX}, width: 640, height: 480, fps: 30, rotation: 0}, wrist: {type: opencv, index_or_path: ${WRIST_IDX}, width: 640, height: 480, fps: 30}}" \
    --policy_host=127.0.0.1 \
    --policy_port=6667 \
    --action_horizon=8 \
    --lang_instruction="pick up the the red cube and put into box"
