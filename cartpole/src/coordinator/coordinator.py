import os
import time
import enum
import copy
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

from src.hp_student.agents.ddpg import DDPGAgent
from src.logger.logger import Logger, plot_trajectory
from src.utils.utils import ActionMode, energy_value, logger
from src.physical_design import MATRIX_P

np.set_printoptions(suppress=True)


class Coordinator:

    def __init__(self):

        # Real time status
        self._plant_action = 0
        self._action_mode = ActionMode.STUDENT
        self._last_action_mode = None

    def reset(self, state, epsilon=1):
        self._plant_action = 0
        energy = energy_value(state=state, p_mat=MATRIX_P)
        self._action_mode = ActionMode.STUDENT if energy < epsilon else ActionMode.TEACHER
        print(f"energy is: {energy}")
        print(f"action mode: {self._action_mode}")
        self._last_action_mode = None

    def get_terminal_action(self, hp_action, ha_action, plant_state, epsilon=1, dwell_flag=False):
        self._last_action_mode = self._action_mode

        # Display current system status based on energy
        energy = energy_value(plant_state, MATRIX_P)
        if energy < epsilon:
            logger.debug(f"current system energy status: {energy} < {epsilon}, system is safe")
        else:
            logger.debug(f"current system energy status: {energy} >= {epsilon}, system is unsafe")

        # When Teacher disabled or deactivated
        if ha_action is None:
            logger.debug("HA-Teacher is deactivated, use HP-Student's action instead")
            self._action_mode = ActionMode.STUDENT
            self._plant_action = hp_action
            return hp_action, ActionMode.STUDENT

        # Teacher is activated
        if self._last_action_mode == ActionMode.TEACHER:
            # Teacher Dwell time
            if dwell_flag is True:
                logger.debug("Continue HA-Teacher action in dwell time")
                self._action_mode = ActionMode.TEACHER
                self._plant_action = ha_action
                return ha_action, ActionMode.TEACHER

            # Switch back to HPC
            else:
                self._action_mode = ActionMode.STUDENT
                self._plant_action = hp_action
                logger.debug(f"Max HA-Teacher dwell time achieved, switch back to HP-Student control")
                return hp_action, ActionMode.STUDENT

        elif self._last_action_mode == ActionMode.STUDENT:

            # Inside safety envelope (bounded by epsilon)
            if energy < epsilon:
                self._action_mode = ActionMode.STUDENT
                self._plant_action = hp_action
                logger.debug(f"Continue HP-Student action")
                return hp_action, ActionMode.STUDENT

            # Outside safety envelope (bounded by epsilon=1)
            else:
                logger.debug(f"Switch to HA-Teacher action for safety concern")
                self._action_mode = ActionMode.TEACHER
                self._plant_action = ha_action
                return ha_action, ActionMode.TEACHER
        else:
            raise RuntimeError(f"Unrecognized last action mode: {self._last_action_mode}")

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode
