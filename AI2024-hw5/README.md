# Homework 5

## Install Necessary Packages
conda create -n hw5 python=3.11 -y
conda activate hw5
pip install -r requirements.txt

# execute the training and evaluation of your code.
python pacman.py
python pacman.py --eval --eval_model_path submissions/pacma_dqn.p

# my environmnet
Ubuntu 22.04.3 LTS

# plot the picture
conda install pandas
run the plot_file.ipynb file