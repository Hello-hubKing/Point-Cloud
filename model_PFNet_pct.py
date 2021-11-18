#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##全局CBAM注意力
##gcn 编码（有问题）一层训练，效果明显降低
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import  numpy as np

#knn
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=2, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
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

    return feature
class Point_Transformer_Last(nn.Module):
    def __init__(self, channels):
        super(Point_Transformer_Last, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x
# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)#计算两点之间的欧氏距离
    # print(sqrdists)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)#在dim维度上选取前nsample
    return group_idx
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
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


def farthest_point_sample(xyz, npoint, RAN=True):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if RAN:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()

    fps_idx = farthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)#返回xyz与new_xyz之间距离最近的nsample个点[B, S, nsample]

    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.k = point_scales
        self.point_scales = point_scales
        self.ca = ChannelAttention(1920)
        self.sa = SpatialAttention()

        # self.conv_gcn1=torch.nn.Conv2d(6,64,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)



        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last_1 = Point_Transformer_Last(128)
        self.pt_last_2= Point_Transformer_Last(256)

        self.maxpool_128=torch.nn.MaxPool2d((1,512),1)
        self.maxpool_256 = torch.nn.MaxPool2d((1, 256),1)


    def forward(self, x):
        #PCT
        xyz = x
        x=x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=2, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)

        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=2, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x_1 = self.pt_last_1(feature_0)
        x_2 = self.pt_last_2(feature_1)

        #未加入归一化层
        x_1 = torch.cat([x_1, feature_0], dim=1)
        x_2 = torch.cat([x_2, feature_1], dim=1)



        x_1=x_1.unsqueeze(2)
        x_1=self.maxpool_128(x_1)
        x_1 = x_1.squeeze(2)

        x_2 = x_2.unsqueeze(2)
        x_2 = self.maxpool_256(x_2)
        x_2 = x_2.squeeze(2)
        L=[x_1,x_2]
        x = torch.cat(L, 1)
        x=x.unsqueeze(2)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x=x.squeeze(2)
        return x


class Latentfeature(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.ca1 = ChannelAttention(1920)
        self.ca2 = ChannelAttention(1920)
        self.ca3 = ChannelAttention(1920)

    def forward(self, x):

        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        outs1=torch.unsqueeze(outs[0],2)
        outs2=torch.unsqueeze(outs[1],2)
        outs3=torch.unsqueeze(outs[2],2)
        out1=self.ca1(outs1)*outs1
        out1=self.sa1(outs1)*outs1
        out2=self.ca2(outs2)*outs2
        out2=self.sa2(outs2)*outs2
        out3=self.ca3(outs3)*outs3
        out3=self.sa3(outs3)*outs3
        out1=torch.squeeze(out1,2)
        out2=torch.squeeze(out2,2)
        out3=torch.squeeze(out3,2)
        latentfeature = torch.cat(outs, 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))

        latentfeature = torch.squeeze(latentfeature, 1)
        #        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))
        #        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
        #        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
        #        latentfeature = latentfeature + latentfeature_64
        #        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
        #        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
        #        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
        #        latentfeature = latentfeature + latentfeature_256
        #        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
        #        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
        #        latentfeature = self.maxpool(latentfeature)
        #        latentfeature = torch.squeeze(latentfeature,2)

        return latentfeature,out1,out2,out3


class PointcloudCls(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, k=40):
        super(PointcloudCls, self).__init__()
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class _netG(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
        super(_netG, self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1_1 = nn.Linear(1920, 1024)

        self.fc1_1_2 = nn.Linear(1920, 1024)
        self.fc2_1_2 = nn.Linear(1024, 512)

        self.fc1_1_3 = nn.Linear(1920, 1024)
        self.fc2_1_3 = nn.Linear(1024, 512)
        self.fc3_1_3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)

        #        self.bn1 = nn.BatchNorm1d(1024)
        #        self.bn2 = nn.BatchNorm1d(512)
        #        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
        #        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
        #        self.bn5 = nn.BatchNorm1d(64*128)
        #
        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !

    #        self.bn1_ = nn.BatchNorm1d(512)
    #        self.bn2_ = nn.BatchNorm1d(256)

    def forward(self, x):
        x,x_t1,x_t2,x_t3 = self.latentfeature(x)
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        x_t1=x_t1.squeeze()
        x_t1= F.relu(self.fc1_1_1(x_t1))
        x_1=torch.add(x_1,x_t1)

        x_t2=x_t2.squeeze()
        x_t2= F.relu(self.fc1_1_2(x_t2))
        x_t2 = F.relu(self.fc2_1_2(x_t2))
        x_2=torch.add(x_2,x_t2)

        x_t3=x_t3.squeeze()
        x_t3= F.relu(self.fc1_1_3(x_t3))
        x_t3 = F.relu(self.fc2_1_3(x_t3))
        x_t3 = F.relu(self.fc3_1_3(x_t3))
        x_3=torch.add(x_3,x_t3)

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 64, 3)  # 64x3 center1

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 128, 64)
        pc2_xyz = self.conv2_1(pc2_feat)  # 6x64 center2

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 128, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.crop_point_num / 128), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)

        return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,fine


class _netlocalD(nn.Module):
    def __init__(self, crop_point_num):
        super(_netlocalD, self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.avgpool = torch.nn.AvgPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_64 = self.maxpool(x_64)
        x_128 = self.maxpool(x_128)
        x_256 = self.maxpool(x_256)
        x_64 = torch.squeeze(torch.squeeze(x_64, -1), -1)
        x_128 = torch.squeeze(torch.squeeze(x_128, -1), -1)
        x_256 = torch.squeeze(torch.squeeze(x_256, -1), -1)

        Layers = [x_256, x_128, x_64]
        x = torch.cat(Layers, 1)

        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    input1 = torch.randn(64, 2048, 3)
    input2 = torch.randn(64, 512, 3)
    input3 = torch.randn(64, 256, 3)
    input_ = [input1, input2, input3]
    netG = _netG(3, 1, [2048, 512, 256], 1024)
    output = netG(input_)
    print(output)
# !/usr/bin/env python3
# -*- coding: utf-8 -*-



