#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from einops import rearrange

from model_utils import Norm, BiBlock_fp, BiDeformableblock
import pdb

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

        
class DeformablePoTR_seg(nn.Module):
    def __init__(self, args, seg_num_all, num_heads=8):
        super(DeformablePoTR_seg, self).__init__()
        self.k = args.k
        self.seg_num_all =seg_num_all
        
        self.blocks = nn.ModuleList()
        self.fp = nn.ModuleList()
        
        # elif args.size == "middle_loc":
        embed_dims = [32,64,128]

        self.use_LGPA = [True, True, True]
        self.use_DGPA = [True, True, True]
        self.use_att = [True, True, True]
        self.use_upLGPA = [True, True, True]
        
        heads1 = [16,16,16]
        heads2 = [16,16,16]         

        self.gamma = [args.gamma, args.gamma, args.gamma]       

        self.n_blocks = len(embed_dims)
        self.fp_in_dim1, self.fp_in_dim2, self.fp_out_dim, self.num_g = [],[],[],[]

        for i in range(self.n_blocks):
            if i==0:
                self.fp_in_dim1.append(embed_dims[-(i+1)])
            else:
                self.fp_in_dim1.append(embed_dims[-(i+1)])
            if i==0:
                self.fp_in_dim2.append(embed_dims[-(i+1)]*2)
            else:
                self.fp_in_dim2.append(embed_dims[-(i+1)]*2)
            
            if i==self.n_blocks-1:
                self.fp_out_dim.append(128)
            else:
                self.fp_out_dim.append(embed_dims[-(i+1)])
            self.num_g.append(args.num_g//(2**i))



        print("Lin")
        self.emb = nn.Linear(3, embed_dims[0])
        self.init_norm = Norm(embed_dims[0], "BN")
            

        for i in range(self.n_blocks):
            self.blocks.append(BiDeformableblock(args,embed_dims[i],num_heads=heads1[i], qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                                        drop_path=args.droppath, s_ratio=args.s_ratio, k=self.k, use_DGPA=self.use_DGPA[i], use_LGPA=self.use_LGPA[i],
                                        gamma=self.gamma[i], num_g=self.num_g[i], att_type=args.att_type))
                 
            self.fp.append(BiBlock_fp(args, self.fp_in_dim1[i], self.fp_in_dim2[i], self.fp_out_dim[i], num_heads=heads2[i], qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                                    drop_path=args.droppath, use_att=self.use_att[i], use_loc=self.use_upLGPA[i],  gamma=self.gamma[-(i+1)], num_g=self.num_g[-(i+1)], k=self.k, 
                                    tau=args.tau, drop_points=args.drop_points, att_type=args.att_type))
        
        self.cls_linear = nn.Sequential(
            nn.Linear(16,64),
            Norm(64, "BN"),
            nn.ReLU(inplace=True)
        )


        self.fc = nn.Sequential(nn.Linear(self.fp_out_dim[-1]+64, 128),
                                Norm(128, "BN"),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(128, seg_num_all))        

    def forward(self, x, cls_label, get_global=False):
        x = rearrange(x, 'b c n -> b n c')
        b, n, _ = x.shape
        g, g_xyz = None, None
    
        gs=[]
        g_xyzs=[]
        xs, xyzs = [], []
        attmap1, attmap2, attmap3 = [], [], []
        xyz=x
        xyzs.append(xyz)
        x = F.relu(self.init_norm(self.emb(x)))
        xs.append(x)
        
        for i in range(self.n_blocks):
            x, xyz, g, g_xyz, cen, att1, att2, att3= self.blocks[i](x, xyz)
            attmap1.append(att1)
            attmap2.append(att2)
            attmap3.append(att3)
            xyzs.append(xyz)
            xs.append(x)
            gs.append(g)
            g_xyzs.append(g_xyz)

            
        for i in range(self.n_blocks-1):
            x, g, g_xyz, att1, att2, att3  = self.fp[i](xyzs[-i-2], xyzs[-i-1], xs[-i-2], xs[-i-1])
            attmap1.append(att1)
            attmap2.append(att2)
            attmap3.append(att3)
            g_xyzs.append(g_xyz)
            xs[-i-2] = x

        x, g, g_xyz, att1, att2, att3 = self.fp[-1](xyzs[0], xyzs[1], xs[0], xs[1])
        attmap1.append(att1)
        attmap2.append(att2)
        attmap3.append(att3)
        g_xyzs.append(g_xyz)

        cls_label = self.cls_linear(cls_label)
        cls_label = cls_label.view(b, 1, 64).repeat(1,x.shape[1],1)  #b,n, 64
        x = torch.cat((x, cls_label), -1)
        x= self.fc(x)    # b n s
        x = rearrange(x, 'b n s -> b s n')

        if get_global==True:
            return x, g_xyzs, xyzs, attmap1 ,attmap2, attmap3
        else:
            return x


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x