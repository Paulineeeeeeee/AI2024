#!/bin/bash

python main.py \
    --exp_name "${1}" \
    --model_name "unsloth/tinyllama-bnb-4bit" \
    --train \
    --wandb_token "8e8d379e0925dfef642c370d17e1f58218595566" \
    --train_batch_size "${2}" \
    --eval_batch_size  "${3}"\
    --gradient_accumulation_steps "${4}" \
    --lr "${5}" \
    --lr_scheduler_type "${6}" \
    --optimizer "${7}" \
    --weight_decay "${8}" \
    --max_grad_norm "${9}" \
    --warmup_ratio "${10}" \
    --num_epochs "${11}" \