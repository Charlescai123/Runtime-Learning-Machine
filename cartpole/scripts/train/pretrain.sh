#!/bin/bash

# Train
ID="Pretrain"
MODE='train'
CHECKPOINT=null
TEACHER_ENABLE=false
TEACHER_CORRECT=false
DOMAIN_RANDOM_FRICTION_CART=true

APPLY_UNKNOWN_DISTRIBUTION=false
WITH_FRICTION=true
FRICTION_CART=3
FRICTION_POLE=0

RANDOM_SEED=0
TRAINING_BY_STEPS=true
MAX_TRAINING_STEPS=1e5
TRAIN_RANDOM_RESET=true
EVAL_RANDOM_RESET=true
GAMMA=0.8

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_steps=${MAX_TRAINING_STEPS} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.friction_pole=${FRICTION_POLE} \
  cartpole.random_reset.seed=${RANDOM_SEED} \
  cartpole.random_reset.train=${TRAIN_RANDOM_RESET} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  cartpole.domain_random.friction_cart.apply=${DOMAIN_RANDOM_FRICTION_CART} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  ha_teacher.teacher_correct=${TEACHER_CORRECT} \
  hp_student.phydrl.gamma=${GAMMA} \
  hp_student.agents.unknown_distribution.apply=${APPLY_UNKNOWN_DISTRIBUTION}

