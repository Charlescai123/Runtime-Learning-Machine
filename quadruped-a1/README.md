# Quadruped-A1

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.5.0-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)
![Pybullet](https://img.shields.io/badge/Pybullet-3.2.6-brightgreen)

---

This repo proposes the implementation for the paper **Runtime-LearningMachine** to
provide lifetime safety for real robot during RL-based deployment and real-world learning.
(Narration: [video1](https://www.youtube.com/shorts/vJKpNzPLPoE)
and [video2](https://www.youtube.com/watch?v=ZNpJULgLnh0))


[//]: # (## Table of Content)

[//]: # ()

[//]: # (* [Code Structure]&#40;#code-structure&#41;)

[//]: # (* [Environment Setup]&#40;#environment-setup&#41;)

[//]: # (* [PhyDRL Runtime]&#40;#phydrl-runtime&#41;)

[//]: # (* [Running Convex MPC Controller]&#40;#running-convex-mpc-controller&#41;)

[//]: # (    * [In Simulation]&#40;#in-simulation&#41;)

[//]: # (    * [In Real A1 Robot]&#40;#in-real-a1-robot&#41;)

[//]: # (* [Trouble Shootings]&#40;#trouble-shootings&#41;)

---

## Code Structure

The configuration settings are under the folder `config`, and the parameters are spawn in
hierarchy for each instance:

[//]: # (```)

[//]: # (├── config                                            )

[//]: # (├── examples                                )

[//]: # (│      ├── a1_exercise_example.py                     <- Robot makes a sinuous move)

[//]: # (│      ├── a1_sim_to_real_example.py                  <- Robot sim-to-real &#40;for testing&#41;)

[//]: # (│      ├── a1_mpc_controller_example.py               <- Running MPC controller in simulator/real plant)

[//]: # (│      ├── main_drl.py                                <- Training A1 with PhyDRL)

[//]: # (│      └── main_mpc.py                                <- Testing trained PhyDRL policy)

[//]: # (├── locomotion)

[//]: # (│      ├── gait_scheduler                            )

[//]: # (│           ├── gait_scheduler.py                     <- An abstract class)

[//]: # (│           └── offset_gait_scheduler.py              <- Actual gait generator)

[//]: # (│      ├── ha_teacher       )

[//]: # (│           ├── ...)

[//]: # (│           └── ha_teacher.py                         <- HA Teacher   )

[//]: # (│      ├── mpc_controllers                      )

[//]: # (│           ├── mpc_osqp.cc                           <- OSQP library for stance state controller)

[//]: # (│           ├── qp_torque_optimizer.py                <- QP solver for stance acceleration controller)

[//]: # (│           ├── stance_leg_controller_mpc.py          <- Stance controller &#40;objective func -> state&#41;)

[//]: # (│           ├── stance_leg_controller_quadprog.py     <- Stance controller &#40;objective func -> acceleration&#41;)

[//]: # (│           └── swing_leg_controller.py               <- Swing controller &#40;using Raibert formula&#41;)

[//]: # (│      ├── robots)

[//]: # (│           ├── ...)

[//]: # (│           ├── a1.py                                 <- A1 robot &#40;for simulation&#41;)

[//]: # (│           ├── a1_robot.py                           <- A1 robot &#40;for real plant&#41;)

[//]: # (│           ├── a1_robot_phydrl.py                    <- A1 robot &#40;for PhyDRL training&#41;)

[//]: # (│           ├── motors.py                             <- A1 motor model)

[//]: # (│           └── quadruped.py                          <- An abstract base class for all robots)

[//]: # (│      ├── state_estimators)

[//]: # (│           ├── a1_robot_state_estimator.py           <- State estimator for real A1)

[//]: # (│           ├── com_velocity_estimator.py             <- CoM velocity estimator simulator/real plant )

[//]: # (│           └── moving_window_fillter.py              <- A filter used in CoM velocity estimator)

[//]: # (│      ├── wbc_controller.py                          <- robot whole-body controller)

[//]: # (│      └── wbc_controller_cl.py                       <- robot whole-body controller &#40;For continual learning&#41;)

[//]: # (├── ...)

[//]: # (├── logs                                              <- Log files for training)

[//]: # (├── models                                            <- Trained model saved path)

[//]: # (├── third_party                                       <- Code by third parties &#40;unitree, qpsolver, etc.&#41;)

[//]: # (├── requirements.txt                                  <- Depencencies for code environment)

[//]: # (├── setup.py)

[//]: # (└── utils.py                         )

[//]: # (```)

[//]: # (## Running Convex MPC Controller:)

[//]: # ()

[//]: # (### Setup the environment)

[//]: # ()

[//]: # (First, make sure the environment is setup by following the steps in the [Setup]&#40;#Setup&#41; section.)

[//]: # ()

[//]: # (### Run the code:)

[//]: # ()

[//]: # (```bash)

[//]: # (python -m src.convex_mpc_controller.convex_mpc_controller_example --show_gui=True --max_time_secs=10 --world=plane)

[//]: # (```)

[//]: # ()

[//]: # (change `world` argument to be one of `[plane, slope, stair, uneven]` for different worlds. The current MPC controller)

[//]: # (has been tuned for all four worlds.)

## Environment Setup

### Setup for Local PC

It is recommended to create a separate virtualenv or conda environment to avoid conflicting with existing system
packages. The required packages have been tested under Python 3.8, though they should be compatible with other Python
versions.

Follow the steps below to build the Python environment:

1. First, install all dependent packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. Second, install the C++ binding for the convex MPC controller:

   ```bash
   python setup.py install
   ```

3. Lastly, build and install the interface to Unitree's SDK. The
   Unitree [repo](https://github.com/unitreerobotics/unitree_legged_sdk) keeps releasing new SDK versions. For
   convenience, we have included the version that we used in `third_party/unitree_legged_sdk`.

   First, make sure the required packages are installed, following
   Unitree's [guide](https://github.com/unitreerobotics/unitree_legged_sdk?tab=readme-ov-file#dependencies). Most
   nostably, please make sure to
   install `Boost` and `LCM`:

   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   ```

   Then, go to `third_party/unitree_legged_sdk` and create a build folder:
   ```bash
   cd third_party/unitree_legged_sdk
   mkdir build && cd build
   ```

   Now, build the libraries and move them to the main directory by running:
   ```bash
   cmake ..
   make
   mv robot_interface* ../../..
   ```

### Setup for Real Robot

Follow the steps if you want to run controllers on the real robot:

1. **Setup correct permissions for non-sudo user (optional)**

   Since the Unitree SDK requires memory locking and high process priority, root priority with `sudo` is usually
   required to execute commands. To run the SDK without `sudo`, write the following
   to `/etc/security/limits.d/90-unitree.conf`:

   ```bash
   <username> soft memlock unlimited
   <username> hard memlock unlimited
   <username> soft nice eip
   <username> hard nice eip
   ```

   Log out and log back in for the above changes to take effect.

2. **Connect to the real robot**

   Configure the wireless on the real robot with the [manual](docs/A1_Wireless_Configuration.pdf), and make sure
   you can ping into the robot's low-level controller (IP:`192.168.123.10`) on your local PC.

### Runtime

To test it in simulation, run `bash scripts/locomotion_test.sh`. To test it in hardware, make sure the parameters are
well set and run `bash scripts/a1_test.sh`.
