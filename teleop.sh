#!/usr/bin/env bash
# Joint-only web teleop for the PLAIN so100_follower (no IK / no EE wrapper).
#
# Opens http://localhost:8080/ ; press-and-hold each motor's +/- button to jog it
# (target ramps while held, holds when released). STOP releases all axes; HOME
# drives every motor to its calibrated center (0 deg). The two robot cameras
# stream live at the top of the same panel.
#
# Cameras (SO-100 layout): top/front = "USB Camera", wrist = "Innomaker-U20CAM".
# macOS shuffles OpenCV indices whenever devices come/go (iPhone Continuity
# Camera, replugs), so resolve them by hardware fingerprint at launch instead of
# hardcoding. Both 640x480 @ 30fps.
IDX=($(.venv/bin/python resolve_cameras.py front wrist)) || exit 1
TOP_IDX=${IDX[0]}
WRIST_IDX=${IDX[1]}
echo "Resolved camera indices: top=${TOP_IDX} wrist=${WRIST_IDX}"

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
lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem5AE60583301 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: ${TOP_IDX}, width: 640, height: 480, fps: 30, rotation: 0}, wrist: {type: opencv, index_or_path: ${WRIST_IDX}, width: 640, height: 480, fps: 30}}" \
    --teleop.type=web_so100 \
    --teleop.host=127.0.0.1 \
    --teleop.port=8080 \
    --display_data=false
