# Environment Setup

This release branch keeps the environments used by the Milestone-GAE paper:
ALFWorld and SciWorld.

## ALFWorld

Install ALFWorld and its text-world dependencies:

```bash
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0
pip install alfworld
```

Download the ALFWorld game files:

```bash
alfworld-download -f
```

The training scripts read the data path from `ALFWORLD_DATA`:

```bash
export ALFWORLD_DATA=$HOME/.cache/alfworld
```

## SciWorld

SciWorld requires Java 1.8 or newer. The source tree includes the ScienceWorld
Python package and its JAR used by the paper experiments.

```bash
cd agent_system/environments/env_package/sciworld/ScienceWorld
pip install -e .
cd -
```

The release scripts use generalization level 1:

```text
env.sciworld.generalization_level=1
```
