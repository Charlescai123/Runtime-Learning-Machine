# RLM-Quadruped-Go2

![PyTorch](https://img.shields.io/badge/PyTorch-3.2.6-red?logo=pytorch)
![Tensorflow](https://img.shields.io/badge/Tensorflow-2.11.0-orange?logo=tensorflow)
![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-darkgrey?logo=isaacgym)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)

---

This repo deploys the **Runtime-LearningMachine** on Quadruped-Go2 robot to within a sim2sim environment (Pybullet to
IsaacGym)

## User Guide

### Dependencies

* *Python - 3.8 above*
* *PyTorch - 1.10.0*
* *Isaac Gym - Preview 4*

### Setup

1. Clone this repository:

```bash
git clone git@github.com:Charlescai123/isaac-phyrl-go2.git
```

2. Create the conda environment with:

```bash
conda env create -f environment.yml
```

3. Activate conda environment and Install `rsl_rl` lib:

```bash
conda activate phyrl-go2
cd extern/rsl_rl && pip install -e .
```

4. Download and install IsaacGym:

* Download [IsaacGym](https://developer.nvidia.com/isaac-gym) and extract the downloaded file to the root folder.
* navigate to the `isaacgym/python` folder and install it with commands:
* ```bash
  cd isaacgym/python && pip install -e .
  ```
* Test the given example (ignore the error about GPU not being utilized if any):
* ```bash
  cd examples && python 1080_balls_of_solitude.py
  ```
