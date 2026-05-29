# lerobot_robot_so_follower_ee

Auto-discovery shim that registers a single-follower **keyboard end-effector** robot
type with lerobot. Lets `lerobot-teleoperate` drive an SO-100/101 follower from the
keyboard alone (no leader arm), with arrow keys / Shift / Ctrl producing Cartesian
EE deltas that are converted to joint targets via inverse kinematics.

Real implementation lives at `src/lerobot/robots/so_follower_ee/` (additive in-tree
module). This dist exists only so that lerobot's `register_third_party_plugins()`
discovers and imports the side-effect registration.

## Install

```bash
uv pip install -e plugins/lerobot_robot_so_follower_ee
```

The dist depends on `placo` (kinematics solver) plus macOS-specific `cmeel` pins
needed because the prebuilt placo wheel links against urdfdom v4 / tinyxml2 v10
while the cmeel default is v6 / v11:

```toml
"placo>=0.9.6,<0.9.17"
"cmeel-urdfdom<5"
"cmeel-tinyxml2<11"
```

On Linux the upstream cmeel chain should resolve cleanly without these pins.

## Run

```bash
sh teleop.sh
```

which expands to:

```bash
lerobot-teleoperate \
    --robot.type=so100_follower_ee \
    --robot.port=/dev/tty.usbmodem5AE60583301 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=keyboard_ee \
    --display_data=false
```

## Keys

| Key       | Effect                                                             |
| --------- | ------------------------------------------------------------------ |
| ← / →     | ±X (step size from `end_effector_step_sizes`, default 2 cm/tick)   |
| ↑ / ↓     | ∓Y                                                                 |
| Shift     | −Z   /   Shift_R: +Z                                               |
| Ctrl_L    | close gripper   /   Ctrl_R: open gripper                           |
| ESC       | disconnect & quit                                                  |

## Config knobs

Field defaults live in
`src/lerobot/robots/so_follower_ee/config_so_follower_ee.py`. Override on the CLI:

- `--robot.urdf_path=...` — by default points at the local ManiSkill clone
  (`/Users/dongmao.zhang/upstream/ManiSkill/mani_skill/assets/robots/so100/so100.urdf`).
- `--robot.target_frame_name=...` — defaults to `Fixed_Jaw_tip` (URDF link).
- `--robot.end_effector_step_sizes={"x":0.02,"y":0.02,"z":0.02}`
- `--robot.end_effector_bounds={"min":[-0.5,-0.5,-0.1],"max":[0.5,0.5,0.7]}`
- `--robot.gripper_speed_factor=20.0`

## How it works

`lerobot-teleoperate` calls `make_robot_from_config()` which constructs
`SOFollowerEndEffector`. The robot internally builds a 5-step processor pipeline
the first time `connect()` succeeds:

```
keyboard_ee output {delta_x, delta_y, delta_z, gripper}
  → KeyboardDeltaToEECmd (bridge: emit {enabled, target_x/y/z, target_wx/wy/wz, gripper_vel})
  → EEReferenceAndDelta (FK current joints → apply delta → ee.x/y/z/wx/wy/wz)
  → EEBoundsAndSafety (clip to workspace)
  → GripperVelocityToJoint (discrete keyboard {0,1,2} → integrated ee.gripper_pos)
  → InverseKinematicsEEToJoints (IK → six {motor}.pos)
  → SOFollower.send_action (writes joint targets to the Feetech bus)
```

`use_latched_reference=False` so holding a key keeps producing fresh deltas from
the current measured pose, instead of latching one reference at key-down.
