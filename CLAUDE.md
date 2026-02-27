# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a Hugging Face library for real-world robotics in PyTorch. It provides tools for training and deploying robot learning policies, managing datasets, and controlling physical robots. Source code lives in `src/lerobot/`, installed as an editable package.

## Common Commands

### Installation
```bash
pip install -e ".[dev,test]"
pre-commit install
git lfs install && git lfs pull  # needed for test artifacts
```

### Linting & Formatting
```bash
pre-commit run --all-files       # all checks (ruff, typos, bandit, mypy, gitleaks, pyupgrade, prettier)
ruff check --fix src/            # lint with auto-fix
ruff format src/                 # format code
```

### Testing
```bash
pytest tests/ -sv                          # full test suite
pytest tests/test_specific.py -sv          # single test file
pytest tests/test_specific.py::test_fn -sv # single test function
make test-end-to-end DEVICE=cpu            # end-to-end policy train+eval (ACT, Diffusion, TDMPC, SmolVLA)
```

Test device defaults to `cuda` if available, override with `LEROBOT_TEST_DEVICE=cpu`.

### CLI Entry Points
All CLI commands are defined in `pyproject.toml [project.scripts]`:
- `lerobot-train` — Train a policy
- `lerobot-eval` — Evaluate a policy
- `lerobot-record` — Record robot demonstrations
- `lerobot-teleoperate` — Teleoperate a robot
- `lerobot-calibrate` — Calibrate robot motors
- `lerobot-dataset-viz` — Visualize dataset
- `lerobot-info` — Show dataset/model info
- `lerobot-edit-dataset` — Edit dataset metadata
- `lerobot-find-cameras` / `lerobot-find-port` / `lerobot-find-joint-limits` — Hardware discovery
- `lerobot-setup-motors` — Motor setup wizard
- `lerobot-replay` — Replay recorded episodes
- `lerobot-train-tokenizer` — Train a tokenizer for tokenizer-based policies
- `lerobot-imgtransform-viz` — Visualize image transforms

CLI arguments use `draccus` (dataclass-based arg parsing) with dot-notation:
```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --batch_size=8
```

## Architecture

### Configuration System
Configs are Python dataclasses parsed by `draccus` (not YAML). The hierarchy:
- `configs/train.py` → `TrainPipelineConfig` (top-level: dataset, policy, env, optimizer, scheduler, eval, wandb, peft)
- `configs/eval.py` → `EvalPipelineConfig` (standalone eval pipeline config)
- `configs/policies.py` → `PreTrainedConfig` (base policy config, uses `draccus.ChoiceRegistry` for polymorphic dispatch via `--policy.type=<name>`)
- `configs/default.py` → `DatasetConfig`, `EvalConfig`, `WandBConfig`, `PeftConfig`
- `configs/types.py` → Core enums (`FeatureType`, `NormalizationMode`) and `PolicyFeature` dataclass
- `configs/parser.py` → Custom draccus parsing helpers for CLI argument handling

Config loading from Hub or local path uses `draccus.config_type("json")` to parse saved `train_config.json` / `config.json`. Resume training loads config from checkpoint: `--config_path=<checkpoint>/pretrained_model/train_config.json --resume=true`.

### Policy Pattern
Each policy in `policies/<name>/` follows a consistent three-file structure:
- `configuration_<name>.py` — Config dataclass, registered via `@PreTrainedConfig.register_subclass("<name>")`
- `modeling_<name>.py` — `nn.Module` implementation with `forward()`, `select_action()`, and normalization logic
- `processor_<name>.py` — Pre/post processing (normalization, image transforms); factory function `make_<name>_pre_post_processors()`

Policies are discovered at import time via the registration decorator. The `PreTrainedConfig` base class provides HuggingFace Hub integration (`from_pretrained`/`save_pretrained`).

`policies/factory.py` is the central entry point: `get_policy_class()` uses lazy imports to avoid loading all policies at startup. `make_policy()` wires up config → features → model instantiation. `make_pre_post_processors()` creates normalization pipelines from dataset stats. The factory also supports 3rd-party policy plugins via dynamic import fallback (`_get_policy_cls_from_policy_name`).

Available policies: `act`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi0_fast`, `pi05`, `sac`, `smolvla`, `groot`, `wall_x`, `xvla`, `sarm`. SAC also registers a `reward_classifier` subclass for its reward model. `rtc` (Real Time Chunking) is a config mixin in `policies/rtc/` used by pi0-family policies, not a standalone policy.

### Training Pipeline
`scripts/lerobot_train.py` → `train()` function:
1. Parses `TrainPipelineConfig` via draccus
2. Creates dataset via `datasets/factory.py` → `make_dataset()` (returns `LeRobotDataset`)
3. Creates policy via `policies/factory.py` → `make_policy()` (infers features from dataset/env)
4. Creates optimizer/scheduler via `optim/factory.py` → `make_optimizer_and_scheduler()`
5. Wraps everything with `Accelerator` for distributed training / mixed precision
6. Training loop: `update_policy()` does forward/backward/clip/step
7. Periodic eval via `lerobot_eval.py:eval_policy_all()` in sim environments
8. Checkpoints saved to `output_dir/checkpoints/{step}/pretrained_model/`

### Hardware Abstraction
Robots, cameras, motors, and teleoperators each have their own subdirectory under `src/lerobot/` with a base class and per-hardware implementations:
- `robots/robot.py` — Base robot interface
- `cameras/` — OpenCV, Intel RealSense, ZMQ, Reachy2
- `motors/` — Dynamixel, Feetech
- `teleoperators/` — Gamepad, keyboard, phone, leader-follower arms

### Datasets
`datasets/lerobot_dataset.py` contains `LeRobotDataset`, which wraps Parquet (structured data) + MP4/images (visual data) with HuggingFace Hub push/pull. Dataset format is v3.0 (`datasets/v30/`).

Dataset on-disk layout:
```
dataset_name/
├── meta/info.json              # features, fps, robot_type, splits
├── meta/stats.json             # per-feature min/max/mean/std for normalization
├── meta/tasks.parquet          # task_index → natural language description
├── data/chunk-000/file-000.parquet   # frame-level: action, state, timestamps, indices
└── videos/{camera_key}/chunk-000/file-000.mp4
```

### Optional Dependencies
Hardware and policy extras are defined in `pyproject.toml [project.optional-dependencies]`. Install what you need:
```bash
pip install -e ".[dynamixel,feetech,aloha,pusht,smolvla]"
```

Note: `pi` extra requires a special transformers fork. `groot` requires flash-attn with specific install steps. `wallx` pins exact versions of transformers/peft.

### Processor Pipeline
`processor/` is a generic, sequential data processing framework for transforming robotics data (observations, actions, rewards). Key types:
- `ProcessorStep` — Abstract base class for a single transformation, registered via `ProcessorStepRegistry`
- `DataProcessorPipeline` — Chains multiple `ProcessorStep` instances; integrates with HuggingFace Hub for sharing/versioning
- Core type aliases in `processor/core.py`: `PolicyAction` (tensor), `RobotAction` (dict), `EnvAction` (ndarray), `RobotObservation` (dict), `EnvTransition` (TypedDict)

Specialized steps: `NormalizeProcessor`, `DeviceProcessor`, `ObservationProcessor`, `BatchProcessor`, `DeltaActionProcessor`, `TokenizerProcessor`, `HilProcessor`, `RenameProcessor`, `GymActionProcessor`, `EnvProcessor`. Each policy's `make_<name>_pre_post_processors()` factory assembles the appropriate pipeline.

### Additional Modules
- `async_inference/` — Async policy inference with a client-server split: `policy_server.py` runs the model, `robot_client.py` handles robot I/O. Uses gRPC via `transport/` for communication.
- `rl/` — Online reinforcement learning infrastructure (SAC): `learner.py`/`learner_service.py` for training, `actor.py` for rollouts, `buffer.py` for replay, `gym_manipulator.py` for Gym env wrappers.
- `transport/` — gRPC service definitions (`services.proto` → `services_pb2.py`) for distributed inference between policy server and robot client.
- `data_processing/` — Offline data processing utilities (e.g., SARM annotation tools).

## Code Style

- Python 3.10+, line length 110, double quotes
- Ruff for linting (E, W, F, I, B, C4, T20, N, UP, SIM) and formatting
- mypy enabled for `configs`, `envs`, `optim`, `model`, `transport`, `cameras` modules (strict for `configs`); excludes `tests/`, `examples/`, `benchmarks/`
- Pre-commit also runs: typos, bandit (security), gitleaks (secret detection), zizmor (GitHub Actions), pyupgrade (py310+), prettier (markdown)
- Generated protobuf files (`*_pb2.py`, `*_pb2_grpc.py`) are excluded from linting

## Test Structure

Tests mirror the source layout under `tests/` (e.g., `tests/policies/`, `tests/datasets/`, `tests/robots/`). Shared fixtures are in `tests/fixtures/` and loaded as pytest plugins via `tests/conftest.py`:
- `tests.fixtures.dataset_factories` — Dataset creation helpers
- `tests.fixtures.files` — Temp file management
- `tests.fixtures.hub` — HuggingFace Hub mocking
- `tests.fixtures.optimizers` — Optimizer fixtures

Decorator helpers in `tests/utils.py` (`require_cuda`, `require_env`, `require_package`) handle conditional skipping.
