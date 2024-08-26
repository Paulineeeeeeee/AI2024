# setting 
conda create -y -n ai_hw6 python=3.10
conda activate ai_hw6
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install -U "xformers<0.0.26" --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install tqdm packaging wandb

# grading
# done
bash run.sh DPO unsloth/mistral-7b-v0.3-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566
bash run.sh ORPO unsloth/mistral-7b-v0.3-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566

bash run.sh DPO unsloth/gemma-2b-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566
bash run.sh ORPO unsloth/gemma-2b-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566

bash run.sh DPO unsloth/tinyllama-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566
bash run.sh ORPO unsloth/tinyllama-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566

bash train.sh DPO 4 4 16 1e-5 cosine paged_adamw_32bit 0.01 1 0.1 1
bash train.sh "DPO" 4 4 16 5e-6 "linear" "paged_adamw_8bit" 0.01 1 0.1 3
bash train.sh "DPO" 4 4 16 1e-6 "cosine" "paged_adamw_32bit" 0.01 1 0.1 5
bash train.sh "DPO" 4 4 16 5e-6 "linear" "paged_adamw_32bit" 0.01 1 0.1 3
bash train.sh "DPO" 4 4 16 1e-5 "cosine" "paged_adamw_8bit" 0.01 1 0.1 1


bash train.sh "ORPO" 4 4 16 1e-5 "cosine" "paged_adamw_32bit" 0.01 1 0.1 1
bash train.sh "ORPO" 4 4 16 5e-6 "linear" "paged_adamw_8bit" 0.01 1 0.1 3
bash train.sh "ORPO" 4 4 16 1e-6 "cosine" "paged_adamw_32bit" 0.01 1 0.1 5
bash train.sh "ORPO" 4 4 16 5e-6 "linear" "paged_adamw_32bit" 0.01 1 0.1 3
bash train.sh "ORPO" 4 4 16 1e-5 "cosine" "paged_adamw_8bit" 0.01 1 0.1 1

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


# submission
# done
bash inference.sh  unsloth/mistral-7b-v0.3-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566
bash inference.sh  unsloth/gemma-2b-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566
bash inference.sh  unsloth/tinyllama-bnb-4bit 8e8d379e0925dfef642c370d17e1f58218595566
