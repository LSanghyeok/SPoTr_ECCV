#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_partseg.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from timm.scheduler import create_scheduler
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import _init_path
from seg.data import ShapeNetPart, ShapeNetPart2
from seg.model import DGCNN_partseg, DeformablePoTR_seg
from util import cal_loss, IOStream
import copy
import pdb

global class_cnts
class_indexs = np.zeros((16,), dtype=int)
global visual_warning
visual_warning = True

class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def _init_(rank):
    if rank==0:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('checkpoints/'+args.exp_name):
            os.makedirs('checkpoints/'+args.exp_name)
        if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
            os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
        if not os.path.exists('checkpoints/'+args.exp_name+'/'+'visualization'):
            os.makedirs('checkpoints/'+args.exp_name+'/'+'visualization')
        os.system('cp main_partseg.py checkpoints'+'/'+args.exp_name+'/'+'main_partseg.py.backup')
        os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
        os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
        os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    if not visual:
        label = label.squeeze()
    shape_ious = []
    overall_ious = []
    category = {}    
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
            overall_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

        if label[shape_idx] not in category:
            category[label[shape_idx]] = [shape_ious[-1]]
        else:
            category[label[shape_idx]].append(shape_ious[-1])
    # part_iou = np.mean(part_ioud)
    return shape_ious, category


def visualization(visu, visu_format, data, pred, seg, label, partseg_colors, class_choice):
    global class_indexs
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        classname = class_choices[int(label[i])]
        class_index = class_indexs[int(label[i])]
        if visu[0] != 'all':
            if len(visu) != 1:
                if visu[0] != classname or visu[1] != str(class_index):
                    skip = True 
                else:
                    visual_warning = False
            elif visu[0] != classname:
                skip = True 
            else:
                visual_warning = False
        elif class_choice != None:
            skip = True
        else:
            visual_warning = False
        if skip:
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1
        else:  
            if not os.path.exists('checkpoints/'+args.exp_name+'/'+'visualization'+'/'+classname):
                os.makedirs('checkpoints/'+args.exp_name+'/'+'visualization'+'/'+classname)
            for j in range(0, data.shape[2]):
                RGB.append(partseg_colors[int(pred[i][j])])
                RGB_gt.append(partseg_colors[int(seg[i][j])])
            pred_np = []
            seg_np = []
            pred_np.append(pred[i].cpu().numpy())
            seg_np.append(seg[i].cpu().numpy())
            xyz_np = data[i].cpu().numpy()
            xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
            IoU = calculate_shape_IoU(np.array(pred_np), np.array(seg_np), label[i].cpu().numpy(), class_choice, visual=True)
            IoU = str(round(IoU[0], 4))
            filepath = 'checkpoints/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_pred_'+IoU+'.'+visu_format
            filepath_gt = 'checkpoints/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_gt.'+visu_format
            if visu_format=='txt':
                np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ') 
                np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
                print('TXT visualization file saved in', filepath)
                print('TXT visualization file saved in', filepath_gt)
            elif visu_format=='ply':
                xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath_gt)
                print('PLY visualization file saved in', filepath)
                print('PLY visualization file saved in', filepath_gt)
            else:
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                (visu_format))
                exit()
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1


def train(args, io):
    
    if args.use_sample=="True":
        train_dataset = ShapeNetPart2(split='trainval', aug=args.aug, num_points=args.num_points)
        test_dataset = ShapeNetPart2(split='test', num_points=args.num_points)
        if (len(train_dataset) < 100):
            drop_last = False
        else:
            drop_last = True
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

        train_loader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=drop_last, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, 
                                num_workers=4, batch_size=args.test_batch_size, shuffle=False, drop_last=False, sampler=test_sampler)
 
    elif args.use_sample=="False":
        train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
        test_dataset = ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice)

        if (len(train_dataset) < 100):
            drop_last = False
        else:
            drop_last = True

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

        train_loader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=drop_last, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, num_workers=4, batch_size=args.test_batch_size, shuffle=False, drop_last=False, sampler=test_sampler)
        print("Hello")        
        
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.use_wandb == "True" and args.rank==0:
        wandb.init(project="ECCV2022_rebuttal_seg", entity="mlvpc")
        wandb.config.update(args)
    #Try to load models
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    elif args.model == 'rbf':
        model = DeformablePoTR_seg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")

    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[0], output_device=0, find_unused_parameters=True)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.ema == "True":
        model_ema = copy.deepcopy(model)
        for param in model_ema.parameters():
            param.detach_()
        step = 0
        best_test_iou_ema = 0
        counter = 0
        
    if args.optimizer=="SGD":
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer=="AdamW":
        print("Use Adam")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer=="Adam":
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    # lr_scheduler, _ = create_scheduler(args, opt)

    if args.sched == "cosine":
        lr_scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.min_lr)
    elif args.sched == 'step':
        lr_scheduler = MultiStepLR(opt, [140, 180], gamma=0.1)    
    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        if args.ema=="True":
            model_ema.train()
            
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in tqdm(train_loader):
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if args.ema=="True":
                alpha = min(1 - 1 / (step + 1), args.ema_decay)
                step+=1
                with torch.no_grad():
                    for ema_v, model_v in zip(model_ema.state_dict().values(), model.state_dict().values()):
                        ema_v.copy_(alpha * ema_v + (1-alpha) * model_v)
            
            pred = seg_pred.max(dim=2)[1] # (batch_size, num_points)
            # import pdb
            # pdb.set_trace()
            if args.world_size>1:
                dist.reduce(loss.detach(), dst=0)
                if args.rank==0:
                    label_list = [torch.zeros_like(label).cuda() for _ in range(args.world_size)]
                    seg_list = [torch.zeros_like(seg).cuda() for _ in range(args.world_size)]
                    pred_list = [torch.zeros_like(pred).cuda() for _ in range(args.world_size)]

                    dist.gather(label.cuda(), label_list, dst=0)
                    dist.gather(seg, seg_list, dst=0)
                    dist.gather(pred, pred_list, dst=0)
  
                    label = torch.cat(label_list,0).cpu()
                    seg = torch.cat(seg_list,0)
                    pred = torch.cat(pred_list,0)
                    loss = loss
                else:
                    dist.gather(label, dst=0)
                    dist.gather(seg, dst=0)
                    dist.gather(pred, dst=0)
                    
            if args.rank==0:
                count += batch_size*args.world_size
                train_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
                pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
                train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
                train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
                train_true_seg.append(seg_np)
                train_pred_seg.append(pred_np)
                train_label_seg.append(label.reshape(-1))

        lr_scheduler.step()
        if args.rank==0:
            train_true_cls = np.concatenate(train_true_cls)
            train_pred_cls = np.concatenate(train_pred_cls)
            train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
            train_true_seg = np.concatenate(train_true_seg, axis=0)
            train_pred_seg = np.concatenate(train_pred_seg, axis=0)
            train_label_seg = np.concatenate(train_label_seg)
            train_ious, _ = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                    train_loss*1.0/count,
                                                                                                    train_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(train_ious))
            io.cprint(outstr)
            log_dict = {"Train Loss": train_loss*1.0/count, 
                        "Train Acc": train_acc, 
                        "Train avg Acc": avg_per_class_acc,
                        "Train IoU" : np.mean(train_ious)}
            
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in tqdm(test_loader):
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            with torch.no_grad():
                seg_pred = model(data, label_one_hot)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
                
            pred = seg_pred.max(dim=2)[1]
            
            if args.world_size>1:
                dist.reduce(loss.detach(), dst=0)
                if args.rank==0:
                    label_list = [torch.zeros_like(label).cuda() for _ in range(args.world_size)]
                    seg_list = [torch.zeros_like(seg).cuda() for _ in range(args.world_size)]
                    pred_list = [torch.zeros_like(pred).cuda() for _ in range(args.world_size)]

                    dist.gather(label.cuda(), label_list, dst=0)
                    dist.gather(seg, seg_list, dst=0)
                    dist.gather(pred, pred_list, dst=0)
  
                    label = torch.cat(label_list,0).cpu()
                    seg = torch.cat(seg_list,0)
                    pred = torch.cat(pred_list,0)
                    
                    loss = loss
                else:
                    dist.gather(label, dst=0)
                    dist.gather(seg, dst=0)
                    dist.gather(pred, dst=0)
                    
            if args.rank==0: 
                count += batch_size*args.world_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                test_label_seg.append(label.reshape(-1))
                
        if args.rank==0: 
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_label_seg = np.concatenate(test_label_seg)
            test_ious, category = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
            cat_ious = np.mean(list(category.values()))
            mean_value = []
            for values in category.values():
                mean_value.append(np.mean(values))
            cat_ious = np.mean(mean_value)
            if np.mean(test_ious) >= best_test_iou:
                best_test_iou = np.mean(test_ious)
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)    
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, cat iou: %.6f, test iou: %.6f, Best test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              cat_ious,
                                                                                              np.mean(test_ious),
                                                                                              best_test_iou)
            io.cprint(outstr)
            log_dict2 = {"Test Loss": test_loss*1.0/count, 
                        "Test Acc": test_acc, 
                        "Test avg Acc": avg_per_class_acc,
                        "Cat iou": cat_ious,
                        "Test iou": np.mean(test_ious),
                        "Best Test iou" : best_test_iou}

            log_dict.update(log_dict2)
            
        if args.ema=="True":
            test_loss = 0.0
            count = 0.0

            model_ema.eval()
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            test_label_seg = []
            for data, label, seg in tqdm(test_loader):
                seg = seg - seg_start_index
                label_one_hot = np.zeros((label.shape[0], 16))
                for idx in range(label.shape[0]):
                    label_one_hot[idx, label[idx]] = 1
                label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
                data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                
                with torch.no_grad():
                    seg_pred = model_ema(data, label_one_hot)
                    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                    loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
                    
                pred = seg_pred.max(dim=2)[1]
                
                if args.world_size>1:
                    dist.reduce(loss.detach(), dst=0)
                    if args.rank==0:
                        label_list = [torch.zeros_like(label).cuda() for _ in range(args.world_size)]
                        seg_list = [torch.zeros_like(seg).cuda() for _ in range(args.world_size)]
                        pred_list = [torch.zeros_like(pred).cuda() for _ in range(args.world_size)]
    
                        dist.gather(label.cuda(), label_list, dst=0)
                        dist.gather(seg, seg_list, dst=0)
                        dist.gather(pred, pred_list, dst=0)
    
                        label = torch.cat(label_list,0).cpu()
                        seg = torch.cat(seg_list,0)
                        pred = torch.cat(pred_list,0)
                        
                        loss = loss
                    else:
                        dist.gather(label, dst=0)
                        dist.gather(seg, dst=0)
                        dist.gather(pred, dst=0)
                        
                if args.rank==0: 
                    count += batch_size*args.world_size
                    test_loss += loss.item() * batch_size
                    seg_np = seg.cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    test_true_cls.append(seg_np.reshape(-1))
                    test_pred_cls.append(pred_np.reshape(-1))
                    test_true_seg.append(seg_np)
                    test_pred_seg.append(pred_np)
                    test_label_seg.append(label.reshape(-1))
                    
            if args.rank==0: 
                test_true_cls = np.concatenate(test_true_cls)
                test_pred_cls = np.concatenate(test_pred_cls)
                test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
                test_true_seg = np.concatenate(test_true_seg, axis=0)
                test_pred_seg = np.concatenate(test_pred_seg, axis=0)
                test_label_seg = np.concatenate(test_label_seg)
                test_ious, category = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
                mean_value = []
                for values in category.values():
                    mean_value.append(np.mean(values))
                cat_ious = np.mean(mean_value)                
                if np.mean(test_ious) >= best_test_iou_ema:
                    counter = 0
                    best_test_iou_ema = np.mean(test_ious)
                    torch.save(model_ema.state_dict(), 'checkpoints/%s/models/model_ema.t7' % args.exp_name)    
    
                 
                outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, cat iou: %.6f, test iou: %.6f, Best test iou: %.6f' % (epoch,
                                                                                                test_loss*1.0/count,
                                                                                                test_acc,
                                                                                                avg_per_class_acc,
                                                                                                cat_ious,
                                                                                                np.mean(test_ious),
                                                                                                best_test_iou_ema)
                io.cprint(outstr)
                log_dict3 = {"Test Loss": test_loss*1.0/count, 
                            "Test Acc_ema": test_acc, 
                            "Test avg Acc_ema": avg_per_class_acc,
                            "Cat iou_ema": cat_ious,
                            "Test iou_ema": np.mean(test_ious),
                            "Best Test iou_ema" : best_test_iou_ema}


                log_dict.update(log_dict3)

                counter += 1
                if args.epochs==500 and counter == args.patience:
                    break

        if args.use_wandb == "True":
            wandb.log(log_dict)

def test(args, io):
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    partseg_colors = test_loader.dataset.partseg_colors
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in tqdm(test_loader):
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
        # visiualization
        visualization(args.visu, args.visu_format, data, pred, seg, label, partseg_colors, args.class_choice) 
    if visual_warning and args.visu != '':
        print('Visualization Failed: You can only choose a point cloud shape to visualize within the scope of the test class')
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious, category = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    if np.mean(test_ious) >= best_test_iou:
        best_test_iou = np.mean(test_ious)
        torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)    
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f, Best test iou: %.6f' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious),
                                                                             best_test_iou)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--port', type=int, default=11111)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='rbf', metavar='N',
                        choices=['dgcnn', 'rbf', 'BN'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=999, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=128, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
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
    parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='RATE',
                        help='weight_decay')
    parser.add_argument('--droppath', type=float, default=0, metavar='DP',
                        help='0-1')         
    parser.add_argument("--s_ratio", type=float, default=1/4)
    parser.add_argument("--gamma", type=float, default=3)         
    parser.add_argument("--num_g", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--aug", type=str, default="True")

    parser.add_argument("--min_lr", type=float, default=0.0005)
    parser.add_argument("--drop_points", type=float, default=0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--use_sample", type=str, default="True")
    parser.add_argument("--use_wandb", type=str, default="False")
    parser.add_argument("--ema", type=str, default="True")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--att_type", type=str, default="dot")

    args = parser.parse_args()

    args.warmup_lr = args.lr/1000

    # args.exp_name = args.exp_name + "{}".format(args.seed)

    assert(torch.cuda.device_count() == 1)
    torch.cuda.set_device(0)
    
    url = "tcp://localhost:{}".format(args.port)
    dist.init_process_group("gloo", init_method=url, rank=args.rank, world_size=args.world_size)
    
    _init_(args.rank)

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
