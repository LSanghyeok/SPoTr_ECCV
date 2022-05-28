import subprocess
import argparse
import torch

parser = argparse.ArgumentParser(description="run")
parser.add_argument("--device", type=str)
parser.add_argument("--world_size", type=int)
parser.add_argument("--rank", type=int)
parser.add_argument("--run", type=int)
parser.add_argument("--port", type=int, default = 5678)

args = parser.parse_args()


#base
# num_g = 16
# k =16

# lr 0.05 0.1 tommorw{0.15 0.2}
# min_lr 0.001, 0.0005, 0.0001
# weight_decay 0.0001, 0.0005, 0.001
if args.run == 0:
    seed = torch.randint(0,10000,(1,)).item()
    weight_decay = 0.0001
    exp_name = "exp1"
    epochs =200
    for lr in [0.05, 0.1]:
        min_lr = lr /100
        if args.rank == 0:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --use_wandb True --seed {seed}", shell=True)
        else:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --seed {seed}", shell=True)

if args.run == 1:
    seed = torch.randint(0,10000,(1,)).item()
    weight_decay = 0.0005
    exp_name = "exp2"
    epochs =200
    for lr in [0.05, 0.1]:
        min_lr = lr /100
        if args.rank == 0:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank}  --port {args.port} --use_wandb True --seed {seed}", shell=True)
        else:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --seed {seed}", shell=True)

if args.run == 2:
    seed = torch.randint(0,10000,(1,)).item()
    weight_decay = 0.001
    exp_name = "exp3"
    epochs =200
    for lr in [0.05, 0.1]:
        min_lr = lr /100
        if args.rank == 0:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --use_wandb True --seed {seed}", shell=True)
        else:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --seed {seed}", shell=True)

if args.run == 3:
    seed = torch.randint(0,10000,(1,)).item()
    weight_decay = 0.0001
    exp_name = "exp4"
    epochs =200
    for lr in [0.15, 0.2]:
        min_lr = lr /100
        if args.rank == 0:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --use_wandb True --seed {seed}", shell=True)
        else:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --seed {seed}", shell=True)


if args.run == 4:
    seed = torch.randint(0,10000,(1,)).item()
    weight_decay = 0.0001
    exp_name = "exp5"
    epochs =200
    for lr in [0.15, 0.2]:
        min_lr = lr /100
        if args.rank == 0:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --use_wandb True --seed {seed}", shell=True)
        else:
            subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --exp_name {exp_name} --epochs {epochs} --lr {lr} --min_lr {min_lr} --weight_decay {weight_decay}\
            --world_size {args.world_size} --rank {args.rank} --port {args.port} --seed {seed}", shell=True)
