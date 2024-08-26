#!/bin/bash

bash train.sh DPO 2 2 8 1e-5 cosine paged_adamw_32bit 0 0 0 1
bash train.sh "DPO" 2 2 8 5e-6 "linear" "paged_adamw_8bit" 0 0 0 3
bash train.sh "DPO" 2 2 8 1e-6 "cosine" "paged_adamw_32bit" 0 0 0 5
bash train.sh "DPO" 2 2 8 5e-6 "linear" "paged_adamw_32bit" 0 0 0 3
bash train.sh "DPO" 2 2 8 1e-5 "cosine" "paged_adamw_8bit" 0 0 0 1

bash train.sh "ORPO" 2 2 8 1e-5 "cosine" "paged_adamw_32bit" 0 0 0 1
bash train.sh "ORPO" 2 2 8 5e-6 "linear" "paged_adamw_8bit" 0 0 0 3
bash train.sh "ORPO" 2 2 8 1e-6 "cosine" "paged_adamw_32bit" 0 0 0 5
bash train.sh "ORPO" 2 2 8 5e-6 "linear" "paged_adamw_32bit" 0 0 0 3
bash train.sh "ORPO" 2 2 8 1e-5 "cosine" "paged_adamw_8bit" 0 0 0 1