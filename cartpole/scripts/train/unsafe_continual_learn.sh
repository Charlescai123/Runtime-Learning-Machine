#!/bin/bash

# Train
ID="Unsafe-Continual-Learn"
MODE='train'
CHECKPOINT="results/models/Pretrain"
TEACHER_ENABLE=false
TEACHER_CORRECT=false
WITH_FRICTION=true
FRICTION_CART=30

APPLY_UNKNOWN_DISTRIBUTION=true
RANDOM_RESET_SEED=2
TRAIN_RANDOM_RESET=true
EVAL_RANDOM_RESET=true
ACTUATOR_NOISE=true
#TRAIN_RANDOM_RESET=false
#EVAL_RANDOM_RESET=false

#TRAINING_BY_STEPS=true
#MAX_TRAINING_EPISODES=1e3
TRAINING_BY_STEPS=false
MAX_TRAINING_EPISODES=10
GAMMA=0.8

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_episodes=${MAX_TRAINING_EPISODES} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.random_reset.train=${TRAIN_RANDOM_RESET} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  cartpole.random_reset.seed=${RANDOM_RESET_SEED} \
  cartpole.domain_random.actuator.apply=${ACTUATOR_NOISE} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  ha_teacher.teacher_correct=${TEACHER_CORRECT} \
  hp_student.phydrl.gamma=${GAMMA} \
  hp_student.agents.unknown_distribution.apply=${APPLY_UNKNOWN_DISTRIBUTION}