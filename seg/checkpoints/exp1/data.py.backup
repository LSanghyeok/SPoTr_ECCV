#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
Modified by 
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@Time: 2021/7/20 7:49 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def translate_pointcloud2(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        # self.partseg_colors = load_color_partseg()
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()
    
class ShapeNetPart2():
    def __init__(self, num_points = 2048, split='train', aug="False"):
        self.transforms = transforms.Compose([PointcloudToTensor()])
        self.num_points = num_points
        
        self.aug=aug
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal')
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.split=split
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.seg_num_all = 50
        self.seg_start_index = 0
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k:v for k,v in self.cat.items()}
            
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split=='train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..'%(split))
                exit(-1)
                
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        
        self.cache = {}
        self.cache_size = 20000
    def __len__(self):
        return len(self.datapath)
        
    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]

            point_set = pc_normalize(point_set)
            seg = data[:,-1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, cls)

            if len(seg) > self.num_points:
                choice = np.random.choice(len(seg), self.num_points, replace=False)
            else:
                choice = np.random.choice(len(seg), self.num_points, replace=True)
                #resample
            point_set = point_set[choice, :]
            seg = seg[choice]

        if self.split == 'trainval' and self.aug == "True":
           point_set = translate_pointcloud(point_set)

        if self.transforms is not None:
            point_set = self.transforms(point_set)
        return point_set, torch.from_numpy(cls), torch.from_numpy(seg)

