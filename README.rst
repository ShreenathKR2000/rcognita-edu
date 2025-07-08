===============================
Benchmarking Control Algorithms
===============================

This repository contains the code for benchmarking three control strategies
– Kinematic (Nominal), Linear Quadratic Regulator (LQR), and Model Predictive Control (MPC) – 
for trajectory tracking of a 3‑wheeled mobile robot within the `rcognita-edu` simulation framework.

Project Structure
=================
  
- ``systems.py``  
  Robot models (``Sys3WRobotNI`` for kinematics; ``Sys3WRobot`` for dynamics).  
- ``controllers.py``  
  Implementations of N_CTRL (kinematic), ControllerLQR, ControllerMPC.  
- ``preset_3wrobot_NI.py``  
  Entry point for running benchmarks:  
  - ``--ctrl_mode N_CTRL``  
  - ``--ctrl_mode LQR``  
  - ``--ctrl_mode MPC``  
- ``kinematic_results/``, ``lqr_results/``, ``mpc_results/``  
  Folders where simulation plots are saved.  
- ``launch.bash`` (optional)  
  Convenience script to run all three benchmarks in sequence.

Installation
============

1. Clone this repository:
   ::
   
     git clone https://github.com/ShreenathKR2000/rcognita-edu.git
     cd rcognita-edu

2. (Optional) Create and activate a virtual environment:
   ::
   
     python3 -m venv .venv
     source .venv/bin/activate

3. Install dependencies:
   ::
   
     pip install -r requirements.txt

Usage
=====

To run a specific controller benchmark, use:

::

  python3 preset_3wrobot_NI.py --ctrl_mode N_CTRL
  python3 preset_3wrobot_NI.py --ctrl_mode LQR
  python3 preset_3wrobot_NI.py --ctrl_mode MPC

By default, each script saves three figures (trajectories, errors, costs/controls)
in the respective results folder.

Example: run all three in one go via the provided Bash script
(if you added one):

::

  bash launch.bash

Results
=======

After running, you will find:

- ``kinematic_results/all_trajectories.png``, ``all_errors.png``, ``all_controls.png``
- ``lqr_results/all_trajectories.png``, ``all_errors.png``, ``all_costs.png``,``all_controls.png``
- ``mpc_results/all_trajectories.png``, ``all_errors.png``, ``all_costs.png``,``all_controls.png``

These plots are used in the accompanying benchmarking report.


Repository Links
================

- **This assignment** (benchmarking fork):  
  https://github.com/ShreenathKR2000/rcognita-edu/tree/assignment-benchmarking  

- **Original framework**:  
  https://github.com/thd-research/rcognita-edu  
