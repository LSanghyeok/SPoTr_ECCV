
  
import os
import sys
import copy
import math
import numpy as np
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from timm.models.layers import DropPath


import pdb

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


#Pointnet_Pointnet2_pytorch : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def farthest_point_sample(src, npoint):
    """
    Args:
        src ([b,n,3]): Source points position.
        npoint ([int]): Number of points to sample.
    Returns:
        centroids ([b,npoint]): Sampled points.
    """    

    device = src.device
    B, N, C = src.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = src[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((src - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


#Pointnet_Pointnet2_pytorch : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def index_points(points, idx):
    """
    Args:
        points ([b,n,3]): Source points position.
        idx ([b,m]): Index of points to sample.
    Returns:
        new_points ([b,m,3]): Sampled Points.
    """   
     
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

#Pointnet_Pointnet2_pytorch : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def square_distance(src, dst):
    """
    Args:
        src ([b,n,3]): Source points position.
        dst ([b,m,3]): Target points position.
    Returns:
        dist ([b,n,m]): Point-wise square distance.
    """    
    
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def bikernel(src, dst, src2, dst2, dst_feature, gamma=1.):
    """Kernel function
    
    Args:
        src ([b,n,c]): Source points query.
        dst ([b,m,c]): Target points key.
        dst_feature ([b,m,d]): Target points value.
        gamma (float, optional): kernel bandwidth. Defaults to 1..
        norm (str, optional): Type of normalization. Defaults to "sum".
    Raises:
        NotImplementedError: [description]
    Returns:
        kern_feat ([b,n,d]): Source points value.
    """    
    ver = "rbf"
    # ver="linear"
    # ver = "poly"
    if ver =="rbf":
        dist = square_distance(src, dst)
        if not isinstance(gamma, float) and not isinstance(gamma, int): 
            gamma = gamma.reshape(1, -1, 1)
        dist_exp = (dist * (-gamma)).exp()
        zero = torch.zeros(1, dtype=torch.float32).to(dist_exp.device)
        att_map1 = torch.where(dist_exp <= 0.05, zero, dist_exp)

    elif ver =="linear":   
        att_map1 = einsum('b n d, b m d -> b n m', src, dst)
        zero = torch.zeros(1, dtype=torch.float32).to(att_map1.device)
        att_map1 = torch.where(att_map1 <= 0.05, zero, att_map1)

    elif ver =="poly":
        att_map1 = einsum('b n d, b m d -> b n m', src, dst)**2
        zero = torch.zeros(1, dtype=torch.float32).to(att_map1.device)
        att_map1 = torch.where(att_map1 <= 0.05, zero, att_map1)

    att_map2 = einsum('b n d, b m d -> b n m', src2, dst2)
    att_map2 = att_map2.softmax(dim=-1) #(b, n, m)

    att_map = att_map1 * att_map2
    kern_feat = torch.bmm(att_map, dst_feature)
    return kern_feat, att_map1, att_map2

def kernel(src, dst, dst_feature, gamma=1., norm="sum"):
    """Kernel function
    
    Args:
        src ([b,n,c]): Source points query.
        dst ([b,m,c]): Target points key.
        dst_feature ([b,m,d]): Target points value.
        gamma (float, optional): kernel bandwidth. Defaults to 1..
        norm (str, optional): Type of normalization. Defaults to "sum".
    Raises:
        NotImplementedError: [description]
    Returns:
        kern_feat ([b,n,d]): Source points value.
    """    
    
    if norm == "sum":
        dist = square_distance(src, dst)
        if not isinstance(gamma, float) and not isinstance(gamma, int): 
            gamma = gamma.reshape(1, -1, 1)
        dist_exp = (dist * (-gamma)).exp()
        zero = torch.zeros(1, dtype=torch.float32).to(dist_exp.device)
        att_map = torch.where(dist_exp <= 0.05, zero, dist_exp)
    elif norm == "softmax":
        dist = square_distance(src, dst)
        dist_exp = (dist * (-gamma)).exp()
        att_map = dist_exp/(dist_exp.sum(-1, keepdim=True)+1e-6)
    elif norm =="att":
        att_map = einsum('b n d, b m d -> b n m', src, dst)
        att_map = att_map.softmax(dim=-1) #(b, n, m)
    else:
        raise NotImplementedError

    kern_feat = torch.bmm(att_map, dst_feature)
    return kern_feat

def rel_fps_pos(xyz):
    """Normalize position to center point.
    Args:
        xyz ([b,m,k,3]): Absolute position.
    Returns:
        xyz ([b,m,k,3]): Relative position to anchor.
        dist ([b,m,k]): Distance to anchor.
        centroid_pos ([b,m,3]): anchor point position.
    """    
    
    centroid_pos = xyz[:,:,0:1,:]#(b, m, 1,3)
    xyz = xyz-centroid_pos #(b, m, k, 3)
    dist = xyz.pow(2).sum(-1).unsqueeze(-1) #(b,m,k)
    return xyz, dist, centroid_pos

def knn(xyz, center, k):
    """
    Args:
        xyz ([b,n,3]): Points position.
        center ([b,m,3]): Center point position.
        k (int): Number of neighbors.
    Returns:
        idx ([b,m,k]): Index of nearest neighbors.
    """    
    
    pairwise_dist = -square_distance(center,xyz) #(b,m,n)
    idx = pairwise_dist.topk(k=k, dim=-1)[1] #(b,m,k)

    return idx

def fps_grouping(x, xyz, s_ratio, k):
    """
    Args:
        x ([b,n,c]): Points feature.
        xyz ([b,n,3]): Points position.
        s_ratio (float): Sampling ratio.
        k (int): Number of neighbors.
    Returns:
        x ([b,n,k,c]): Grouped feature.
        x ([b,n,k,3]): Grouped point.
    """    
    
    B,N,_ = xyz.shape
    npoint = int(N * s_ratio)
    fps_idx = farthest_point_sample(xyz, npoint) # (b, npoint)

    new_xyz = index_points(xyz, fps_idx) # (b, npoint, 3)
    idx = knn(xyz, new_xyz, k) #(b, npoint, k)
    
    x = index_points(x, idx) #(b, npoint, k, c)
    xyz = index_points(xyz, idx) #(b, npoint, k, 3)
    return x, xyz

def grouping(x, xyz, k):
    """
    Args:
        x ([b,n,c]): Points feature.
        xyz ([b,n,3]): Points position.
        s_ratio (float): Sampling ratio.
        k (int): Number of neighbors.
    Returns:
        x ([b,n,k,c]): Grouped feature.
        x ([b,n,k,3]): Grouped point.
    """    
    
    device = x.device
    B,N,_ = xyz.shape
    
    # select_idx = torch.arange(N, device=device).long().unsqueeze(0).repeat(B,1)
    # new_xyz = index_points(xyz, select_idx) # (b, npoint, 3)
    idx = knn(xyz, xyz, k) #(b, npoint, k)
    
    x = index_points(x, idx) #(b, npoint, k, c)
    xyz = index_points(xyz, idx) #(b, npoint, k, 3)
    return x, xyz
    
#Pointnet_Pointnet2_pytorch : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        """Feature propagation based on PointNet2

        Args:
            in_dim (int): Input dim.
            out_dim (int): Output dim.
        """        
        super(PointNetFeaturePropagation, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            Norm(out_dim, "BN"),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            Norm(out_dim, "BN"),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1 ([b,n,3]): Skip connected point position.
            xyz2 ([b,s,3]): Previous layer point position.
            points1 ([b,n,out_dim]): Skip connected point feature.
            points2 ([b,s,in_dim]): Previous layer point feature.

        Returns:
            new_points ([b,n,out_dim]): Upsampled point feature.
        """        

        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists=torch.clamp(dists, min=1e-4)

            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-4)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / (norm + 1e-4)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points


        new_points = self.mlp(new_points)
  
        return new_points
    

class Norm(nn.Module):
    def __init__(self, dim, norm_type="LN"):
        """
        Args:
            dim (int): Dim.
            type (str, optional): Type of Normalization ["LN", "BN"]. Defaults to "LN".
        Raises:
            NotImplementedError: 
            
        """
        super(Norm, self).__init__()
        self.type=norm_type
        if norm_type=="LN":
            self.norm = nn.LayerNorm(dim)
        elif norm_type=="BN":
            self.norm = nn.BatchNorm1d(dim)
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x ([...,dim]): Point feature.
        Returns:
            x ([...,dim]): Noramalized point feature.
        """        
        if isinstance(self.norm, nn.LayerNorm):
            return self.norm(x)
        elif isinstance(self.norm, nn.BatchNorm1d):
            if len(x.shape)==3:
                x = self.norm(rearrange(x, 'b n c -> b c n'))
                x = rearrange(x, 'b c n -> b n c')
            elif len(x.shape)==4:
                _, m, k, _ = x.shape
                x = rearrange(x, 'b m k d -> (b m) d k')
                x = self.norm(x)
                x = rearrange(x, '(b m) d k -> b m k d', m=m, k=k)
            return x
            
class BiBlock_fp(nn.Module):
    def __init__(self, args, in_dim1, in_dim2, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 drop_path=0., use_att=True, use_loc=True, gamma=1., num_g=8, k=16, tau=1,  drop_points=0, att_type="mlp"):
        """Feature propagation & DGPA
        Args:
            in_dim1 (int): Input dim of Skip connected point.
            in_dim2 (int): Input dim of Previous layer point.
            out_dim (int): Output dim.
            num_heads (int, optional): Number of heads. Defaults to 8.
            qkv_bias (bool, optional): Using bias for qkv or not. Defaults to False.
            qk_scale (float, optional): Normalization scale for attention map. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weight. Defaults to 0..
            proj_drop (float, optional): Dropout rate for projection weight. Defaults to 0..
            drop_path (float, optional): Dropout rate for path. Defaults to 0..
            use_DGPA (bool, optional): Using DGPA or not. Defaults to True.
            gamma (float, optional): Value for initializing kernel bandwidth. Defaults to 1..
            num_g (int, optional): Number of deformable points. Defaults to 8.
        """        
        super(BiBlock_fp, self).__init__()
        self.use_att = use_att
        self.use_loc = use_loc
        if use_att:
            self.DGPA = Block_DGPA(out_dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, drop_path, gamma, num_g, att_type=att_type)
        if use_loc:
            self.LGPA = Block_LGPA(out_dim, num_heads, k, qkv_bias, qk_scale, attn_drop, proj_drop, drop_path, att_type=att_type)

        self.fp = PointNetFeaturePropagation(args, (in_dim1+in_dim2), out_dim)

        
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1 ([b,n,3]): Skip connected point position.
            xyz2 ([b,s,3]): Previous layer point position.
            points1 ([b,n,in_dim1]): Skip connected point feature.
            points2 ([b,s,in_dim2]): Previous layer point feature.

        Returns:
            x ([b,n,out_dim]): Resulted point feature.
            g ([b,num_g,out_dim]): Global point feature.
            g_xyz ([b,num_g,3]): Global point position.
        """        
        # pdb.set_trace()
        x = self.fp(xyz1, xyz2, points1, points2) #b, n, 3
        if self.use_loc:
            loc_x, xyz, cen = self.LGPA(x, xyz1)
        if self.use_att:
            x, g, g_xyz, att1, att2, att3 = self.DGPA(x, xyz1) # (b, n, c), , 
            if self.use_loc:
                x = loc_x + x
        else:
            g, g_xyz = None, None

        

        #else:
            #g, g_xyz = None, None    
        return x, g, g_xyz, att1, att2 ,att3

class BiDeformableblock(nn.Module):
    def __init__(self, args, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0., 
                 s_ratio=1/4, k=1, use_DGPA=True, use_LGPA=True, L_lgpa=1,  gamma=1., num_g=8, lin_bias=True, tau =1, att_type="mlp"):
        """Block of Deformable-PoTR
        Args:
            dim (int): Dim. [description]
            num_heads (int, optional): Number of heads. Defaults to 8.
            qkv_bias (bool, optional): Using bias for qkv or not. Defaults to False.
            qk_scale (float, optional): Normalization scale for attention map. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weight. Defaults to 0..
            proj_drop (float, optional): Dropout rate for projection weight. Defaults to 0..
            drop_path (float, optional): Dropout rate for path. Defaults to 0..
            s_ratio (float): Sampling ratio.
            k (int): Number of neighbors.
            use_DGPA (bool, optional): Using DGPA or not. Defaults to True.
            use_LGPA (bool, optional): Using LGPA or not. Defaults to True.
            L_lgpa (int, optional): Number of layers of lgpa block
            gamma (float, optional): Value for initializing kernel bandwidth. Defaults to 1..
            num_g (int, optional): Number of deformable points. Defaults to 8.
            lin_bias (bool, optional): Using bias for linear operation or not. Default to True. 
        """        
        super(BiDeformableblock, self).__init__()
        self.use_DGPA = use_DGPA
        self.use_LGPA = use_LGPA
        self.linbnrelu = nn.Sequential(
            nn.Linear(dim, dim*2, bias=lin_bias),
            Norm(dim*2, "BN"),
            nn.ReLU(inplace=True)
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            Norm(dim*2, "BN"),
            nn.ReLU(inplace=True),
            nn.Linear(dim*2, dim*2),
            Norm(dim*2, "BN")
        )
        
        self.LGPA = nn.ModuleList([])
        self.DGPA = nn.ModuleList([])

        for _ in range(L_lgpa):
            if use_LGPA:
                self.LGPA.append(Block_LGPA(dim*2, num_heads, k, qkv_bias, qk_scale, attn_drop, proj_drop, drop_path, lin_bias, att_type=att_type))
            if use_DGPA:
                self.DGPA.append(Block_DGPA(dim*2, num_heads,  qkv_bias, qk_scale, attn_drop, proj_drop, drop_path, gamma, num_g, tau=tau, att_type=att_type))
        
        # self.ffn2 = nn.Sequential(
        #     nn.Linear(dim*2, dim*2*4),
        #     Norm(dim*2*4, "BN"),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim*2*4, dim*2),
        #     Norm(dim*2, "BN")         
        # )
        self.s_ratio = s_ratio
        self.k = k
        self.L_lgpa = L_lgpa
        

    def forward(self, x, xyz):
        
        """
        Args:
            x ([b,n,dim]): Point feature.
            xyz ([b,n,3]): Point position.
        Returns:
            x ([b,m,dim]): Anchor Point feature. (m = n*s_ratio)
            xyz ([b,m,3]): Anchor Point position.
            g ([b,num_g,dim]): Global point feature.
            g_xyz ([b,num_g,3]): Global point position.
            cen ([b,m,3]): Center Point position.
        """        
        # x, xyz = fps_grouping(x, xyz, self.s_ratio, self.k) # (b, n, c) -> (b, m, k, c), m = n*s_ratio
        x = self.linbnrelu(x)
                        
        for l in range(self.L_lgpa):
            if self.use_LGPA:
                loc_x, _, cen = self.LGPA[l](x,xyz)#(b, m, k, c) -> (b,m,2*c), (b, m, k, 3) -> (b,m,3)
            else:
                loc_x, cen = 0, 0
            
            if self.use_DGPA:
                x, g, g_xyz, att1, att2, att3 = self.DGPA[l](x, xyz)
                x = x + loc_x
                    
            else:
                x = loc_x
                g, g_xyz = None, None
        
        x, xyz = fps_grouping(x, xyz, self.s_ratio, self.k) #(b, m, k, c)
        x = x.max(-2).values
        xyz = xyz[:,:,0,:]
        x = F.relu(self.ffn(x) + x)

        # x = F.relu(self.ffn2(x)+x)
        
        return x, xyz, g, g_xyz, cen, att1, att2, att3


class Block_LGPA(nn.Module):
    def __init__(self, dim, num_heads=8, k=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0., lin_bias=True, use_cat=False, att_type="mlp"):
        """Locally-Gropued Point Attention block
        
        Args:
            dim (int): Input dim. [description]
            num_heads (int, optional): Number of heads. Defaults to 8.
            k (int, optional): Number of neighbors, Default to 16.
            qkv_bias (bool, optional): Using bias for qkv or not. Defaults to False.
            qk_scale (float, optional): Normalization scale for attention map. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weight. Defaults to 0..
            proj_drop (float, optional): Dropout rate for projection weight. Defaults to 0..
            drop_path (float, optional): Dropout rate for path. Defaults to 0..
            lin_bias (bool, optional): Using bias for linear operation or not. Default to True. 
            use_cat (bool, optional): Using concatenation for generating attention map.
        """        
        super(Block_LGPA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.k = k
        self.att_type = att_type
        if att_type == "mlp" or att_type == "dot":
            self.att = NewAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, use_cat=use_cat)
            self.lin_cen = nn.Linear(dim, dim, bias=lin_bias)
            self.lin_neigh = nn.Linear(dim, dim, bias=lin_bias)
            self.lin = nn.Linear(dim, dim, bias=lin_bias)
        elif att_type == "dotloc":
            self.att = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        else:
            raise NotImplementedError
        self.pos_emb = nn.Sequential(
            nn.Linear(4, 4),
            Norm(4, "BN"),
            nn.ReLU(inplace=True),
            nn.Linear(4, dim)
        )
   
        self.norm = Norm(dim, "BN")        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, xyz):
        """
        Args:
            x ([b,m,k,dim]): Grouped point feature.
            xyz ([b,m,k,3]): Grouped point position.
        Returns:
            x ([b,m,2*dim]): Anchor point feature.
            xyz ([b,m,3]): Anchor point posiition.
            cen ([b,m,3]): Center point position.
        """        
        prev_x = x #(b,m,d)

        x, xyz = grouping(x, xyz, self.k)
        b,m,k,_ = x.shape

        rel_xyz, dist, cen  = rel_fps_pos(xyz) #(b,m,k,3), (b,m,k,1), (b,m,k)


        rel_x = torch.cat((rel_xyz, dist), -1) #(b,m,k,3+1)
        pos_emb_x = self.pos_emb(rel_x) #(b,m,k,d)
        
        x_cen = x[:,:,0:1,:]

        if self.att_type == "mlp" or self.att_type=="dot":
            geom = self.lin_cen(x)-self.lin_neigh(x_cen)
    
            x = self.lin(x)+pos_emb_x
            x = rearrange(x, 'b m k d -> (b m) k d')#((b,m),k,d)
            x,_ = self.att(x, geom, pos=pos_emb_x) #(b,m,d)

        elif self.att_type == "dotloc":
            x = x + pos_emb_x
            x = rearrange(x, 'b m k d -> (b m) k d')
            x_cen = rearrange(x_cen, 'b m k d -> (b m) k d')

            x = self.att(query=x_cen, key=x, value=x)
            x = rearrange(x, '(b m) 1 d -> b m d', b=b)

        x = self.drop_path(self.norm(x)) + prev_x #(b,m, d)
        x = F.relu(x)

        #down sampling
        xyz = cen.squeeze(-2)

        return x, xyz, cen

class Block_DGPA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0., 
                 gamma = 1., num_g=8, tau=1, att_type="mlp"):
        """Deformable Global Point Attention block
        Args:
            dim (int): Dim. [description]
            num_heads (int, optional): Number of heads. Defaults to 8.
            qkv_bias (bool, optional): Using bias for qkv or not. Defaults to False.
            qk_scale (float, optional): Normalization scale for attention map. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weight. Defaults to 0..
            proj_drop (float, optional): Dropout rate for projection weight. Defaults to 0..
            drop_path (float, optional): Dropout rate for path. Defaults to 0..
            gamma (float, optional): Value for initializing kernel bandwidth. Defaults to 1..
            num_g (int, optional): Number of deformable points. Defaults to 8.
        """        
        super(Block_DGPA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        self.att_type = att_type
        self.num_g = num_g
        if att_type == "mlp":
            self.att = NewAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
            self.lin_g = nn.Linear(dim, dim)
            self.lin_all = nn.Linear(dim, dim)
            self.lin = nn.Linear(dim, dim)
        elif att_type == "dot" or att_type == "dotloc":
            self.att = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        else:
            raise NotImplementedError
        self.g = nn.Parameter(torch.randn(num_g, dim))

        
        
        self.pos_emb = nn.Sequential(
            nn.Linear(4, 4),
            Norm(4, "BN"),
            nn.ReLU(inplace=True),
            nn.Linear(4, dim)
        )        
        
        
        # self.g = nn.Parameter(torch.randn(num_g, dim))
        self.gamma = (torch.ones(1)*gamma)
        
        self.norm = Norm(dim, "BN")
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
    def forward(self, x, xyz):
        """
        Args:
            x ([b,n,c]): Point feature.
            xyz ([b,n,3]): Point position.
        Returns:
            x ([b,n,dim]): Resulted Point feature.
            g ([b,num_g,dim]): Global point feature.
            g_xyz ([b,num_g,3]): Global point position.
        """ 
        
        #Update feature of g
        b, _, _ = x.shape #(b, n, d)



        g = repeat(self.g, 'g d -> b g d', b=b)
        g_xyz = kernel(g, x, xyz, None, norm="att")
        g, att_map1, att_map2 = bikernel(g_xyz, xyz, g, x, x, self.gamma.to(g_xyz.device))

        
        rel_xyz = g_xyz[:,None,:,:] - xyz[:,:,None,:].repeat(1,1,self.num_g,1)
        dist = rel_xyz.pow(2).sum(-1).unsqueeze(-1)
        # dist = rel_xyz
        rel_x = torch.cat((rel_xyz, dist), -1) #(b,m,k,3+1)
        pos_emb_x = self.pos_emb(rel_x) #(b,m,k,d)        
        

        if self.att_type == "mlp":
            geom =  self.lin_g(g)[:,None,:,:] - self.lin_all(x)[:,:,None,:]       #(b, n, g, d)
            prev_x = x
            g = self.lin(g[:,None,:,:])+pos_emb_x
            g = rearrange(g, 'b m k d -> (b m) k d')#((b,m),k,d)
    
            x, att_map3 = self.att(g, geom, pos=pos_emb_x) #(b,m,d)
            
            x = self.drop_path(self.norm(x)) + prev_x #(b,m, d)
            x = F.relu(x)
    
            return x, g, g_xyz, att_map1, att_map2, att_map3
        
        elif self.att_type == "dot" or self.att_type=="dotloc":
            prev_x = x # (b, m, d)
            g = g[:, None, :, :] + pos_emb_x # (b, n, g d)
            g = rearrange(g, 'b m k d -> (b m) k d')#((b,n),g,d)
            x = rearrange(x, 'b m d -> (b m) 1 d') #((b,n),1,d)

            x = self.att(query =x, key =g, value = g)
            x = rearrange(x, '(b m) 1 d -> b m d', b=b)
            x = self.drop_path(self.norm(x)) + prev_x #(b,m, d)
            x = F.relu(x)
    
            return x, g, g_xyz, att_map1, att_map2, None





class NewAttention(nn.Module):    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0, dim_value=None, use_cat=True):
        """Attention layer
        Args:
            dim ([type]): Dim. 
            num_heads (int, optional): Number of heads. Defaults to 8.
            qkv_bias (bool, optional): Using bias for qkv or not. Defaults to False.
            qk_scale (float, optional): Normalization scale for attention map. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weight. Defaults to 0..
            proj_drop (float, optional): Dropout rate for projection weight. Defaults to 0..
            dim_value (int, optional): Dim. Defaults to None.
        """        
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.dim_value = default(dim_value, dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_cat = use_cat
        
        if use_cat:
            in_dim = dim*2
        else:
            in_dim = dim
            
        self.mlp = nn.Sequential(
            Norm(in_dim, "BN"),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, dim),
            Norm(dim, "BN"),
            nn.ReLU(inplace=True),
            nn.Linear(dim, self.num_heads)
        )
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            Norm(dim, "BN"),
            nn.ReLU(inplace=True),
            nn.Linear(dim, self.dim_value)
        )
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key, pos=None, value=None, q_xyz=None, k_xyz=None):
        """
        Args:
            query ([b,n,d]): Query point feature.
            key ([b,m,d], optional): Key point feature.
            value ([b,m,d], optional): Value point feature.
            q_xyz ([b,n,3], optional): Query point position.
            k_xyz ([b,m,d], optional): Key point position.
        Returns:
            x ([b,n,d]): Resulted key point feature.
        """        
        b, m, k, d = key.size()
        k = default(key, query) #(b, m, d)
        v = default(value, query) #(b, m, d)
        # v = self.v(v) 
        v = rearrange(v, '(b m) k (h d) -> (b h) m k d', h=self.num_heads, m=m)
        # pdb.set_trace()
        if self.use_cat:
            key = torch.cat((key, pos),-1)
        else:
            key = key + pos
        attn = self.mlp(key) #(b, m, k, h)
        attn = rearrange(attn, 'b m k h -> (b h) m k', h=self.num_heads)
        attn = attn.unsqueeze(-1) 

        
        attn = attn.softmax(dim=-2) #((b,num_head), n, m)
        attn = self.attn_drop(attn) #((b,num_head), n, m)

        x = (attn * v).sum(2)
        # x = einsum('b n m, b m d -> b n d', attn, v) #((b,num_head), n, head_dim)
        x = rearrange(x, '(b h) m d -> b m (h d)', h = self.num_heads) #((b,num_head), n, head_dim)-> (b, n, d)
        x = self.proj(x) #(b,n,d)
        x = self.proj_drop(x) #(b,n,d)
        
        return x, rearrange(attn, '(b h) n m d -> b h n m d', h = self.num_heads)


class Attention(nn.Module):    
    def __init__(self, dim, num_heads=8,  qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0, dim_value=None):
        """Attention layer
        Args:
            dim ([type]): Dim. 
            num_heads (int, optional): Number of heads. Defaults to 8.
            qkv_bias (bool, optional): Using bias for qkv or not. Defaults to False.
            qk_scale (float, optional): Normalization scale for attention map. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weight. Defaults to 0..
            proj_drop (float, optional): Dropout rate for projection weight. Defaults to 0..
            dim_value (int, optional): Dim. Defaults to None.
        """        
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.dim_value = default(dim_value, dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, self.dim_value)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key=None, value=None, q_xyz=None, k_xyz=None):
        """
        Args:
            query ([b,n,d]): Query point feature.
            key ([b,m,d], optional): Key point feature.
            value ([b,m,d], optional): Value point feature.
            q_xyz ([b,n,3], optional): Query point position.
            k_xyz ([b,m,d], optional): Key point position.
        Returns:
            x ([b,n,d]): Resulted key point feature.
        """        
        q = self.q(query) #(b, n, d) 
        k = default(key, query) #(b, m, d)
        v = default(value, query) #(b, m, d)
        k, v = self.k(k), self.v(v)  #(b, m, d)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.num_heads), (q, k, v)) #(b, n, d) -> ((b,num_head), n, head_dim)

        attn = einsum('b n d, b m d -> b n m', q, k) * self.scale #((b*num_head), n, m)


        attn = attn.softmax(dim=-1) #((b,num_head), n, m)
        attn = self.attn_drop(attn) #((b,num_head), n, m)

        x = einsum('b n m, b m d -> b n d', attn, v) #((b,num_head), n, head_dim)
        x = rearrange(x, '(b h) n d -> b n (h d)', h = self.num_heads) #((b,num_head), n, head_dim)-> (b, n, d)
        x = self.proj(x) #(b,n,d)
        x = self.proj_drop(x) #(b,n,d)
        
        return x