import subprocess
import argparse
import torch

parser = argparse.ArgumentParser(description="run")
parser.add_argument("--device", type=str)
parser.add_argument("--run", type=int)
args = parser.parse_args()


if args.run == 0:
    lr = 0.05
    min_lr =0.0005
    weight_decay= 0.0005
    subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay} --att_type dot --epochs 500 --use_wandb Fals0", shell=True)



# if args.run == 1:
#     lr = 0.05
#     min_lr =0.0005
#     weight_decay= 0.0005
#     subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --lr {lr} --min_lr {min_lr} --exp_name poly --weight_decay {weight_decay} --att_type dot --epochs 500 --use_wandb True", shell=True)
