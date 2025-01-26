#!/bin/bash

# Test
ID="Unsafe-Continual-Learn"
MODE='test'
CHECKPOINT="results/models/Unsafe-Continual-Learn"
TEACHER_ENABLE=false
TEACHER_CORRECT=false
WITH_FRICTION=true
FRICTION_CART=30

ACTUATOR_NOISE=true
APPLY_UNKNOWN_DISTRIBUTION=true

PLOT_PHASE=true
PLOT_TRAJECTORY=true
ANIMATION_SHOW=false
LIVE_TRAJECTORY_SHOW=false

ACTUATOR_NOISE=true
EVAL_RANDOM_RESET=false
SAMPLE_POINTS=500
GAMMA=0.8

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  general.max_evaluation_steps=${SAMPLE_POINTS} \
  logger.fig_plotter.phase.plot=${PLOT_PHASE} \
  logger.fig_plotter.trajectory.plot=${PLOT_TRAJECTORY} \
  logger.live_plotter.animation.show=${ANIMATION_SHOW} \
  logger.live_plotter.live_trajectory.show=${LIVE_TRAJECTORY_SHOW} \
  cartpole.domain_random.actuator.apply=${ACTUATOR_NOISE} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  ha_teacher.teacher_correct=${TEACHER_CORRECT} \
  hp_student.phydrl.gamma=${GAMMA} \
  hp_student.agents.unknown_distribution.apply=${APPLY_UNKNOWN_DISTRIBUTION}

