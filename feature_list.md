# Feature List — lerobot-eval × remote vLLM (GR00T) × local LIBERO

> 需求账本（requirements ledger）。任务开始后，需求描述/验收步骤/测试标准不可随意改写；
> 只允许更新每个 feature 的「状态」字段（`completed` / `not_completed`）。

## 总目标 (from CLAUDE.md)
让 `lerobot-eval` 用一个调用**远程 vLLM**(vllm-omni GR00T-N1.7 OpenPI WebSocket API) 的 policy，
在**本地 macOS** 上用**本地 LIBERO**(已做 macOS 适配) 运行评测；环境用 **uv** 管理。

最终目标命令：
```
lerobot-eval \
  --policy.path=vllm_groot_policy \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --policy.n_action_steps=10 \
  --output_dir=./outputs/libero_eval
```

## 关键事实 (调研结论，固定)
- lerobot 入口：`lerobot-eval` → `src/lerobot/scripts/lerobot_eval.py:main`。
- policy 注册：draccus `PreTrainedConfig.register_subclass(...)`；factory 在
  `src/lerobot/policies/factory.py:get_policy_class()` 做动态导入；`__init__.py` 需 import config。
- policy 基类：`src/lerobot/policies/pretrained.py:PreTrainedPolicy`，需实现
  `get_optim_params / reset / forward / predict_action_chunk / select_action`。
- `--policy.path=X` 走 `PreTrainedConfig.from_pretrained(X)`，读取 `X/config.json` 的 `type`。
- env：`--env.type=libero` 已存在（`LiberoEnv` in `src/lerobot/envs/configs.py`），
  `src/lerobot/envs/libero.py` 用 `from libero.libero import benchmark` 创建环境（依赖本地 LIBERO 包）。
- rollout (`lerobot_eval.py:rollout`) 每步：`preprocess_observation` → `add_envs_task`(注入 `task` 语言指令)
  → `env_preprocessor`(LIBERO 默认 `LiberoProcessorStep`) → `preprocessor`(归一化) → `policy.select_action`。
- LIBERO 原始 obs（preprocess 后）：`observation.images.image`/`image2` (float32 CHW, [0,1])，
  `observation.robot_state.{eef.pos/quat/mat, gripper.qpos/qvel, joints.pos/vel}`。
- LIBERO 动作：7 维 `[dx,dy,dz, d(axis-angle)x3, gripper]`，relative(delta) 控制（默认）。
- vLLM GR00T API：WebSocket `ws://host:port/v1/realtime/robot/openpi`，msgpack-numpy 序列化。
  - 握手：server 先发 metadata（`action_horizon`,`action_keys`,`image_resolution`,
    `n_external_cameras`,`needs_wrist_camera`）。
  - 请求 obs：`session_id, embodiment, modality_config, prompt, state{eef_9d,gripper_position,joint_position}, images{external_0,wrist}`，state 为**原始未归一化**。
  - 响应 actions：`dict[key -> ndarray (action_horizon, dim)]`，**绝对**动作（server 端反归一化 + SE(3) 还原）。
  - 客户端参考：`vllm-omni/examples/online_serving/gr00t/{openpi_client.py,parity_eval.py}`。
- 约束：lerobot `requires-python>=3.12`；LIBERO 锁 `>=3.10,<3.11`（已有可用 3.10 venv）。
- 约束：GR00T 真服务需 CUDA GPU；本机 macOS 无法本地起真服务 → 需 mock server 做本地冒烟测试，
  真实推理指向远程 GPU server。

---

## Features

### F1 — uv 单环境：lerobot + 本地 LIBERO + 客户端依赖共存
- 目标：一个 uv 管理的 venv 同时可 `import lerobot`、`from libero.libero import benchmark`、
  `import websockets` 与 msgpack-numpy 客户端依赖。
- 边界：解决 Python 版本冲突——**保持 lerobot 上游代码不改**，改为让本地 LIBERO 适配 Python 3.12
  （放宽 LIBERO 的 requires-python；robomimic 的 egl-probe 用 uv `--overrides` 在非 Linux 丢弃）。
  venv 位于 `lerobot/.venv`（Python 3.12）；lerobot 与 LIBERO 均 editable 安装。
- 验收：
  1. `<venv>/bin/python -c "import lerobot; from libero.libero import benchmark; import websockets, msgpack; print('ok')"` 成功。
  2. `<venv>/bin/lerobot-eval --help` 正常输出。
- 备注：另需 env-only 修复 `transformers==5.3.0`（5.9 使 PretrainedConfig 变 dataclass，触发 lerobot
  原有 groot 的 import 崩溃；5.3.0 可正常 import，不改 lerobot 源码）。
- 状态：completed

### F2 — 注册新 policy 类型 `vllm_groot`（独立插件包，零改 lerobot）
- 目标：新增**独立可安装包** `lerobot_policy_vllm_groot`（lerobot 第三方插件命名约定，前缀
  `lerobot_policy_`），含 `configuration_vllm_groot.py / modeling_vllm_groot.py / processor_vllm_groot.py`。
  config 用 `@PreTrainedConfig.register_subclass("vllm_groot")`；类名/模块名遵循约定
  （`VllmGrootConfig`→`VllmGrootPolicy`，`configuration_`→`modeling_`/`processor_`），
  由 lerobot `register_third_party_plugins()` 在 `lerobot-eval main()` 中自动 import 注册。
  **不修改 lerobot 任何既有文件**（含 factory/__init__/原有 policy）。
- 边界：config 持有远程连接与编码参数（host/port/path、embodiment、modality_config、
  image_resolution、action_keys、n_action_steps、image_flip、gripper/eef 转换参数等）。
- 验收：
  1. `get_policy_class("vllm_groot")` 返回 `VllmGrootPolicy`。
  2. `"vllm_groot" in PreTrainedConfig` 的注册表（draccus choice）。
  3. 单测：构造 `VllmGrootConfig()` 不报错，字段默认值合理。
- 状态：completed（插件 editable 安装；register_third_party_plugins 后 get_policy_class("vllm_groot")→VllmGrootPolicy）

### F3 — 观测编码器：lerobot batch → OpenPI 请求
- 目标：把 lerobot 的 obs（eef pos/旋转、gripper、joint、images、task 文本）编码为
  GR00T OpenPI 请求 dict（eef_9d=xyz+rot6d，gripper_position，joint_position，images external_0/wrist，prompt）。
- 边界：图像 float CHW[0,1] → uint8 HWC + 可配置 180° 翻转 + resize 到 server 分辨率；
  旋转矩阵→rot6d；gripper qpos(2)→单值；prompt 取自 `observation["task"]`。
- 验收：单测：喂入合成的 libero 风格 batch，产出可被 msgpack 序列化、字段/形状符合 schema 的请求。
- 状态：completed（encoding.py + modeling._build_request；mock 往返成功）

### F4 — 远程客户端：WebSocket + msgpack 调用
- 目标：实现握手→发送 obs→接收 actions 的客户端逻辑，含连接复用/超时/错误处理。
- 边界：复用 `parity_eval._send_inference` 的协议；msgpack-numpy 序列化。
- 验收：对 mock server（F8）发起一次往返，返回符合 `action_keys`/`action_horizon` 的 dict。
- 状态：completed（client.py GrootOpenPIClient.infer/handshake；reduced e2e 验证）

### F5 — 动作解码器：server 绝对动作 → LIBERO 7 维动作 + 队列
- 目标：把 server 返回的绝对 action dict 转为 LIBERO 7 维动作序列；维护 `n_action_steps` 队列，
  `select_action` 每步弹一个。
- 边界：实现一种合理的绝对→LIBERO(delta/absolute) 映射（eef_9d→pos+axis-angle，gripper→[-1,1]）；
  映射策略与控制模式（relative/absolute）保持一致、可配置。
- 验收：单测：合成 action dict → 产出形状 `(n_action_steps, 7)`；连续 `select_action` 行为正确（用完再请求）。
- 状态：completed（mock 往返验证：select_action→(1,7) 有限且∈[-1,1]，predict_action_chunk→(1,10,7)）

### F6 — env 预处理接线（vllm_groot 专用）
- 目标：保证 policy 能拿到构造请求所需的观测。
- 决策变更（零改 lerobot）：不自定义 env 处理链（那需改 lerobot）。改为复用默认
  `LiberoProcessorStep` 产出的 8 维 state `[eef_pos(3), eef_axisangle(3), gripper(2)]` + 双图 + task，
  在编码器内重建 `eef_9d`(xyz+rot6d) 与 `gripper_position`；`joint_position` 暂填 0（caveat）。
- 验收：集成：policy `select_action` 收到 `observation.state`/`observation.images.*`/`task` 并跑通。
- 状态：completed（reduced e2e 已证明 policy 正确消费这些 key）

### F7 — `vllm_groot_policy/` 配置目录（支持 `--policy.path=vllm_groot_policy`）
- 目标：在仓库提供 `vllm_groot_policy/config.json`（`type=vllm_groot` + 连接/编码默认值），
  使目标命令 `--policy.path=vllm_groot_policy` 可被 `PreTrainedConfig.from_pretrained` 加载。
- 边界：`VllmGrootPolicy.from_pretrained` 需重写为「只读 config、不加载 safetensors」。
- 验收：
  1. `PreTrainedConfig.from_pretrained("vllm_groot_policy")` 得到 `VllmGrootConfig`。
  2. `make_policy` 能据此实例化 `VllmGrootPolicy`（features 由 env 提供）。
- 状态：completed（from_pretrained 加载为 VllmGrootConfig；含 policy_preprocessor/postprocessor.json，
  eval 的 device_processor/rename_observations_processor overrides 通过校验）

### F8 — 本地 mock GR00T OpenPI server（无 GPU 冒烟）
- 目标：实现一个轻量 WebSocket server，复现握手 metadata + 返回 dummy 绝对 actions，
  供 F4/F9 在 macOS 本地无 GPU 测试协议链路。
- 验收：启动后，`openpi_client.py` 或本项目客户端能完成一次握手 + 往返且校验通过。
- 状态：completed（`lerobot_policy_vllm_groot.mock_server`；GrootOpenPIClient 往返通过）

### F9 — 端到端本地冒烟测试
- 目标：对 mock server，运行**目标命令**（libero_object, batch_size=1, n_episodes=2,
  n_action_steps=10）完成评测并写出 `outputs/libero_eval/eval_info.json`。
- 验收：命令成功退出（exit 0），生成 eval_info.json，包含 per_task/overall 指标结构。
- 状态：completed（精确目标命令 exit 0；10 任务×2 集=20 episodes；outputs/libero_eval/eval_info.json
  含 per_task/per_group/overall；pc_success=0.0 属预期，因 mock 返回 no-op 动作）

### F10 — 使用文档（手动验证步骤）
- 目标：维护一份可执行的使用说明（启动远程/mock server、运行 eval、参数说明、排错），
  确保人工手动验证步骤始终可执行（对应 CLAUDE.md README 维护要求）。
- 验收：按文档步骤可从零跑通 F9 的本地冒烟。
- 状态：completed（`vllm_groot_plugin/README.md` + `scripts/generate_policy_dir.py`）
