# Runtime-Learning-Machine

![Embodied AI](https://img.shields.io/badge/Embodied%20AI-robot_intelligent-lightgray?style=flat-square&logo=robotframework)
![Physical-AI](https://img.shields.io/badge/Physics--AI-physical--model--guided-yellow?style=flat-square&logo=scikitlearn)
![Cyber-Physical System](https://img.shields.io/badge/Cyber--Physical%20System-CPS-blue?style=flat-square&logo=cloudsmith)


This repository contains the experiments of the **Runtime Learning Machine**, where the framework is validated upon
three different physical systems in different environments. Respectively, they're:

* [Cart-Pole](./cartpole/) in Openai Gym
* [Quadruped-A1](./quadruped-a1/) in Real-world Environment
* [Quadruped-Go2](./quadruped-go2/) in Nvidia IsaacGym

We believe this framework contributes to the advancement of Physical-AI, which enables the development of Embodied
Artificial Intelligence within Cyber-Physical Systems.

![Image](https://github.com/user-attachments/assets/0b6c1b53-0b05-47a6-aaee-3fe4acbd0501)

---

# Cart-Pole

Data-driven methods are prone to the unknown unknowns in the environment which brings safety challenges for the
safety-critical systems. Runtime Learning Machine is designed to address related concerns:

<p align="center">
 <img src="./cartpole/docs/no-rlm.gif" height="265" alt="no-rlm"/>
 <img src="./cartpole/docs/rlm.gif" height="265" alt="rlm"/>
 <br><b>Fig. In unknown environment, DRL agent violates safety constraint (Left) while Runtime Learning Machine enables 
agent to safely learn (right)</b>
</p>

# Quadruped-A1

Runtime Learning Machine on A1 Robot to address unknown unknowns:

https://github.com/user-attachments/assets/0591deaa-65f1-4f43-9c77-0c1c3018929a

# Quadruped-Go2

A Sim-to-Sim policy transfer from Pybullet A1 robot to IsaacGym Go2 robot. With **Runtime Learning Machine**
architecture to address real-time safety concern from unknown environment.

<p align="center">
 <img src="./quadruped-go2/docs/rlm_go2_push.gif" height="450" alt="ani_pretrain"/> 
 <br><b>Fig. Safety Performance of Runtime Learning Machine under Random Push</b>
</p>

# Citation

