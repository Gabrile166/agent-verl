# SciWorld Pipeline Integration — Progress & Handoff Doc

> Created: 2026-02-28  
> Purpose: Handoff document for continuing SciWorld integration work

---

## 1. What Has Been Done (✅ Completed)

### 1.1 Static Milestone Templates Made Optional
- **File**: `rlvmr/milestone/templates.py`
- Accepts `"none"` as `fallback_template` to skip static milestone loading
- SciWorld uses dynamic milestone generation, no static templates needed

### 1.2 `run_sciworld.sh` Script Created
- **File**: `examples/milestone_gae_trainer/run_sciworld.sh`
- Based on `run_alfworld.sh`, adapted for SciWorld
- Key configs: `algorithm.expert.enable=true`, `algorithm.milestone_gae.fallback_template=none`

### 1.3 Hydra Config Fixed
- **File**: `verl/trainer/config/ppo_trainer.yaml`
- Added `sciworld` section to `env` config to fix `Key 'sciworld' is not in struct` error

### 1.4 `SciWorldEnvironmentManager.reset()` Interface Fixed
- **File**: `agent_system/environments/env_manager.py`
- Added `kwargs=None` parameter to match `rollout_loop.py` call signature

### 1.5 Expert Worker Obs Filtering (N+1 Architecture) Fixed
- **File**: `agent_system/environments/env_package/sciworld/envs.py`
- **Root cause**: `SciWorldMultiProcessEnv` returned observations for ALL workers (policy + expert), but pipeline expected policy-only
- **Fix**: Added filtering in `step()`, `reset()`, `get_available_actions`, `get_possible_actions`, `get_admissible_commands`
- `step()` now accepts policy-only actions (128), internally expands to full (144) with dummy expert actions, then filters results back to policy-only
- `reset()` resets all workers but returns only policy worker observations
- Mirrors ALFWorld's `AlfworldEnvs` filtering pattern exactly

---

## 2. Current Status — Pipeline Runs But Hits Resource Limit

### What Works
- ✅ Hydra config loads correctly
- ✅ SciWorld environments initialize (Java JVMs launch)
- ✅ Validation phase completes successfully (128 trajectories, 16 queries)
- ✅ Validation metrics computed: `val/success_rate ≈ 0.05-0.09`
- ✅ Expert worker observations correctly filtered

### What Fails
The pipeline crashes when entering **training** (after validation succeeds):

```
RuntimeError: can't start new thread
→ DataLoader worker exited unexpectedly
```

**Root Cause**: System thread/process limit exhaustion. With `train_batch_size=16` and `rollout.n=8`:
- 16 groups × (8 policy + 1 expert) = **144 JVM processes** + 144 Python worker processes
- Plus Ray, WandB (Go process), vLLM, DataLoader workers
- Total easily exceeds OS `ulimit -u` (max user processes)

### WandB Crash (Separate Issue)
WandB's Go backend (`wandb-core`) crashes, leaving its Unix Socket broken. When Python SDK tries to log metrics → `RuntimeError: unable to perform operation on <UnixTransport closed=True>`.
- **Workaround**: Set `trainer.logger='[console]'` or `pkill -9 -f wandb-core` before running
- **Root cause**: Likely residual wandb-core processes from previous runs conflicting

---

## 3. Short-Term Fix (User Has Applied)

Reduce environment instances in `run_sciworld.sh`:
```bash
data.train_batch_size=8   # was 16
env.rollout.n=4           # was 8
```
This reduces JVM count from 144 → ~45. Awaiting result.

---

## 4. Long-Term Optimization Plan (NOT YET IMPLEMENTED)

### Problem: 1 JVM per Worker
Current architecture in `envs.py::_worker()`:
```python
from scienceworld import ScienceWorldEnv
env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
# ↑ This calls py4j launch_gateway() → spawns a NEW JVM process
```

Each of the 144 workers spawns its own JVM via py4j's `launch_gateway()`. Each JVM is ~200-500MB RAM + many threads.

### Proposed Solution: Shared JVM per Group

**Key insight from code analysis**: `ScienceWorldEnv.__init__` uses py4j to:
1. `launch_gateway()` → spawns JVM + returns port
2. `JavaGateway(port=port)` → connects Python to that JVM
3. `self.server = self._gateway.jvm.scienceworld.runtime.pythonapi.PythonInterface()` → creates an instance

**`PythonInterface()` is a per-instance object**. Multiple instances CAN coexist in one JVM.

#### Architecture Change:
```
CURRENT:  Worker0 → JVM0,  Worker1 → JVM1, ... Worker8 → JVM8   (9 JVMs per group)
PROPOSED: Worker0 → JVM_shared ← Worker1, ..., Worker8 → JVM_shared  (1 JVM per group)
```

#### Implementation Steps:
1. **Modify `SciWorldMultiProcessEnv.__init__`**: Launch 1 JVM per group (instead of per worker)
2. **Pass JVM port to workers**: Workers connect to existing JVM via `JavaGateway(port=shared_port)` instead of calling `launch_gateway()`
3. **Each worker creates its own `PythonInterface()`** on the shared JVM for state isolation
4. **Modify `ScienceWorldEnv.__init__`**: Add option to accept an existing gateway port instead of launching a new JVM

#### Resource Reduction:
- From **144 JVMs** → **16 JVMs** (one per group)
- ~90% reduction in JVM processes and memory

#### Risks:
- Thread safety: Multiple `PythonInterface` instances in one JVM may have shared state in Scala code
- Need to verify Scala `PythonInterface` is truly thread-safe
- py4j gateway thread pool may bottleneck with many concurrent requests

#### Alternative (Heavier):
Modify the Scala `PythonInterface` to support multiple environments per instance (environment slots). Much more code change, requires Scala modifications + JAR rebuild.

---

## 5. Key Files Reference

| File | Role |
|---|---|
| `agent_system/environments/env_package/sciworld/envs.py` | SciWorld multiprocess env, worker process, expert filtering |
| `agent_system/environments/env_manager.py` | `SciWorldEnvironmentManager` — high-level env manager |
| `agent_system/environments/env_package/sciworld/ScienceWorld/scienceworld/scienceworld.py` | py4j JVM bridge (`ScienceWorldEnv` class) |
| `agent_system/multi_turn_rollout/rollout_loop.py` | Rollout orchestration, calls `envs.reset(kwargs=...)` |
| `verl/trainer/config/ppo_trainer.yaml` | Hydra config, `env.sciworld` section |
| `examples/milestone_gae_trainer/run_sciworld.sh` | Shell script with all SciWorld training configs |
| `rlvmr/milestone/templates.py` | Milestone templates, `"none"` support |

---

## 6. How ALFWorld Handles Expert Filtering (Reference)

ALFWorld's `AlfworldEnvs` (in `agent_system/environments/env_package/alfworld/envs.py`):
- Uses Ray actors instead of multiprocessing
- Tracks `self.policy_indices` and `self.expert_indices`
- `reset()` and `step()` filter results to return only policy workers
- `step()` fills dummy actions (`"look"`) for expert workers
- SciWorld now mirrors this pattern exactly

---

## 7. Commands for the Server

```bash
# Kill all residual processes before running
pkill -9 -f wandb-core
pkill -9 -f java
ray stop --force

# Run SciWorld training
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/agent-verl
bash examples/milestone_gae_trainer/run_sciworld.sh

# Check thread/process limits
ulimit -u     # max user processes
ulimit -u 65535  # increase if needed
```
之前跑 ALFWorld 不崩，现在跑 SciWorld 一直崩，根本原因是资源压力完全不同：

ALFWorld vs SciWorld 资源对比
ALFWorld	SciWorld
每个环境实例	轻量 Python 对象	一个完整的 Java JVM 进程
144 个 worker	~144 个 Python 线程（共享内存）	144 个 JVM 进程（每个 ~200-500MB）
额外线程	很少	每个 JVM 内部还有 GC 线程、py4j 网关线程等
总内存占用	~几 GB	几十 GB
总进程/线程数	~几百	上千
WandB 为什么挂
WandB 的 Go 后台进程（wandb-core）通过 Unix Socket 跟 Python SDK 通信。这个 Go 进程：

被 OOM Killer 杀了 — 系统内存被 144 个 JVM 吃光，Linux 的 OOM Killer 优先杀"看起来不重要"的进程，wandb-core 就成了牺牲品
或者线程耗尽连带挂了 — ulimit -u 限制被打满后，Go runtime 的 goroutine 调度器也会异常退出
或者 Socket fd 被回收了 — 文件描述符数量达到上限（ulimit -n），Go 进程的 Socket 被系统强制关闭
之前跑 ALFWorld 时，系统资源很充裕，所以 WandB 从来不会遇到这些问题。SciWorld 把资源压到极限后，WandB 就变成最脆弱的环节了。

根本解决
刚刚加的 try/except 是止血方案（WandB 崩了训练不挂）。真正要解决的是之前讨论的 JVM 共享方案，把 144 个 JVM 降到 16 个，系统资源压力就回到 ALFWorld 的水平了，WandB 自然也不会再崩。