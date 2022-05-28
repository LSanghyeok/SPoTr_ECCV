#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from tqdm import tqdm

from timm.scheduler import create_scheduler

import _init_path
import wandb
from cls.data import ModelNet40, ScanObjectNN
from cls.model import PointNet, DGCNN_cls, DeformablePoTR
from util import cal_loss, IOStream
import copy
from datetime import datetime


def _init_():

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=4,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        args.num_class = 40
    elif args.dataset == 'SONN':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, ver=args.ver, aug=args.aug), num_workers=4,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver=args.ver, aug=args.aug), num_workers=4,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        args.num_class =15
    device = torch.device("cuda" if args.cuda else "cpu")


    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == "rbf":
        model = DeformablePoTR(args).to(device)
    else:
        raise Exception("Not implemented")
    
    if args.ema=="True":
        model_ema = copy.deepcopy(model)
        for param in model_ema.parameters():
            param.detach_()
        step = 0
        best_test_acc_ema = 0
        counter = 0
        val_loss = 100000

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    print("CosineAneelaing")
    if args.use_sgd:
        lr_scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.min_lr)
    else:
        lr_scheduler, _ = create_scheduler(args, opt)
    
    criterion = cal_loss

    if args.use_wandb == "True":
        wandb.init(project="ECCV2022_rebuttal", entity="mlvpc", name=args.exp_name)
        wandb.config.update(args)

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        if args.ema=="True":
            model_ema.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits = model(data)
            loss = criterion(logits, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
                
            if args.ema=="True":
                alpha = min(1 - 1 / (step + 1), args.ema_decay)
                step+=1
                with torch.no_grad():
                    for ema_v, model_v in zip(model_ema.state_dict().values(), model.state_dict().values()):
                        ema_v.copy_(alpha * ema_v + (1-alpha) *model_v)

            # torch.cuda.synchronize()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        if args.use_sgd:
            lr_scheduler.step()
        else:
            lr_scheduler.step(epoch)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        log_dict = {"Train Loss": train_loss*1.0/count, 
                    "Train Acc": metrics.accuracy_score(train_true, train_pred), 
                    "Train avg Acc": metrics.balanced_accuracy_score(train_true, train_pred)}

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            with torch.no_grad():
                logits = model(data)
                loss = criterion(logits, label)

            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, Best test acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              best_test_acc)

        io.cprint(outstr)
        log_dict2 = {"Test Loss": test_loss*1.0/count, 
                    "Test Acc": test_acc, 
                    "Test avg Acc": avg_per_class_acc,
                    "Best Test Acc": best_test_acc}
        log_dict.update(log_dict2)
        
        if args.ema=="True":
            test_loss = 0.0
            count = 0.0
            
            model_ema.eval()
            test_pred = []
            test_true = []
            
            for data, label in tqdm(test_loader):
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                with torch.no_grad():
                    logits = model_ema(data)
                    loss = criterion(logits, label)

                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            if test_acc >= best_test_acc_ema:
                best_test_acc_ema = test_acc
                torch.save(model_ema.state_dict(), 'checkpoints/%s/models/model_ema.t7' % args.exp_name)
                
            if test_loss*1.0/count < val_loss:
                counter=0
                val_loss = test_loss*1.0/count

            outstr = 'Test_ema %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, Best test acc: %.6f' % (epoch,
                                                                                test_loss*1.0/count,
                                                                                test_acc,
                                                                                avg_per_class_acc,
                                                                                best_test_acc_ema)

            io.cprint(outstr)
            log_dict3 = {"Test Loss_ema": test_loss*1.0/count, 
                        "Test Acc_ema": test_acc, 
                        "Test avg Acc_ema": avg_per_class_acc,
                        "Best Test Acc_ema": best_test_acc_ema} 
            
            log_dict.update(log_dict3)
            
            counter += 1
            if args.epochs == 500 and counter == args.patience:
                break
        if args.use_wandb == "True":
            wandb.log(log_dict)
        

class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scales = torch.from_numpy(xyz).float().cuda()
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scales)
        return pc

def test(args, io):
    if args.dataset == "modelnet40":
        args.num_class = 40
        test_loader = DataLoader(ModelNet40(partition='eval', num_points=args.num_points),
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == "SONN":
        args.num_class = 15
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver=args.ver, aug=args.aug), num_workers=4,
                                    batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DeformablePoTR(args).to(device)
    model = nn.DataParallel(model)
    model.module.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    best_oa = 0.0
    best_macc = 0.0
    
    pointscale = PointcloudScale(scale_low=0.8, scale_high=1.18)

    while True:
        test_acc = 0.0
        count = 0.0
        test_true = []
        test_pred = []
        
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            if args.use_voting==True:
                logits=0
                for v in range(10):
                    new_data = data
                    if v > 0:
                        new_data.data = pointscale(new_data.data)
                    with torch.no_grad():
                        logits += F.softmax(model(new_data.permute(0, 2, 1)), dim=1)  # sum 10 preds
                logits /= 10   # avg the preds!
                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

            else:
                with torch.no_grad():
                    logits = model(data.permute(0, 2, 1))
                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)

        if test_acc > best_oa:
            best_oa = test_acc
            best_macc = avg_per_class_acc
            
        print(f"Test OA : {test_acc}  mAcc : {avg_per_class_acc} \nBest OA : {best_oa}  Best mAc : {best_macc}")
        
        log_dict = {"Test acc": test_acc, 
                     "Test avg acc": avg_per_class_acc, 
                     "Best test acc": best_oa,
                     "Best test avg acc": best_macc} 

        
        # f = open("../results/{}.txt".format(args.file), "w")
        # f.write(str(test_acc)+" ")
        # f.write(str(avg_per_class_acc))
        # io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='rbf', metavar='N',
                        choices=['pointnet', 'dgcnn', 'rbf', 'latrbf'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'SONN'])
    parser.add_argument('--ver', type=str, default='hard', metavar='N',
                        choices=['easy', 'hard', 'bg'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=16, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='RATE',
                        help='weight_decay')
    parser.add_argument('--droppath', type=float, default=0, metavar='DP',
                        help='0-1')

    parser.add_argument("--lgpa", type=int, default=1)

    parser.add_argument("--num_g", type=int, default=16)
    parser.add_argument("--min_lr", type=float, default=0.0005)
    parser.add_argument("--aug", type=str, default="jit")
    parser.add_argument("--lin_bias", action='store_true')
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--use_voting", action='store_true')
    parser.add_argument("--ema", type=str, default="True")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--att_type", type=str, default="mlp")

    parser.add_argument('--use_wandb', type=str, default='False', metavar='N')

    args = parser.parse_args()
    args.seed = np.random.randint(1, 10000)
    # args.min_lr = args.lr/10
    args.warmup_lr = args.lr/1000
    args.exp_name = args.exp_name + datetime.now().strftime('-%Y.%m.%d_%H:%M')
    
    _init_()
    
    
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)