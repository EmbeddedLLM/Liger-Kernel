#!/bin/bash

export GPUS_PER_NODE=8
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
export NUM_NODES=1
export WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

export LOCAL_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export LR=1e-5

export MEDUSA_NUM_LAYERS=1
export MEDUSA_HEADS_COEFFICIENT=0.2
export MEDUSA_DECAY_COEFFICIENT=0.8
export MEDUSA_SCHEDULER=constant
export MEDUSA_LR_MULTIPLIER=4.0

export MASTER_ADDR="localhost"
export MASTER_PORT="18800"

# Define the parameter arrays
# medusa_only_heads=(True False)
medusa_only_heads=(False)
use_liger=(True False)
# use_liger=(False)
MEDUSA_NUM_HEADS=(3 5)
# MEDUSA_NUM_HEADS=(5)
model_max_length=(1024 2048 4096 8192 16384 32768)

# Loop through all permutations
for moh in "${medusa_only_heads[@]}"; do
  for ul in "${use_liger[@]}"; do
    for mnh in "${MEDUSA_NUM_HEADS[@]}"; do
      for mml in "${model_max_length[@]}"; do
        # Set the OUTPUT_DIR
        OUTPUT_DIR="llama3-8b-medusa_UseLiger${ul}_medusaheads${moh}_numheads${mnh}_length${mml}"

        # Run the command with the current parameter combination
        accelerate launch --config_file fsdp/acc-fsdp.conf \
          --num_machines $NUM_NODES \
          --num_processes $WORLD_SIZE \
          --main_process_ip $MASTER_ADDR \
          --main_process_port $MASTER_PORT \
          train.py \
          --data_path ./ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
          --bf16 True \
          --output_dir $OUTPUT_DIR \
          --num_train_epochs 10 \
          --per_device_train_batch_size $LOCAL_TRAIN_BATCH_SIZE \
          --per_device_eval_batch_size 1 \
          --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
          --evaluation_strategy "no" \
          --save_strategy "no" \
          --prediction_loss_only \
          --learning_rate $LR \
          --weight_decay 0. \
          --warmup_ratio 0.04 \
          --lr_scheduler_type "cosine" \
          --logging_steps 1 \
          --model_max_length $mml \
          --gradient_checkpointing True \
          --lazy_preprocess False \
          --report_to none \
          --include_num_input_tokens_seen \
          --medusa_num_heads $mnh \
          --medusa_num_layers $MEDUSA_NUM_LAYERS \
          --medusa_heads_coefficient $MEDUSA_HEADS_COEFFICIENT \
          --medusa_decay_coefficient $MEDUSA_DECAY_COEFFICIENT \
          --medusa_scheduler $MEDUSA_SCHEDULER \
          --medusa_lr_multiplier $MEDUSA_LR_MULTIPLIER \
          --medusa_only_heads $moh \
          --medusa_return True \
          --use_liger $ul \
          --torch_compile False

        echo "Completed run with parameters:"
        echo "medusa_only_heads: $moh"
        echo "use_liger: $ul"
        echo "MEDUSA_NUM_HEADS: $mnh"
        echo "model_max_length: $mml"
        echo "OUTPUT_DIR: $OUTPUT_DIR"
        echo "-----------------------------------"
      done
    done
  done
done