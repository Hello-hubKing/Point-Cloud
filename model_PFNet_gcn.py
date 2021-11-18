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

class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.k = point_scales
        self.point_scales = point_scales
        self.ca = ChannelAttention(1920)
        self.sa = SpatialAttention()
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        # self.conv_gcn1=torch.nn.Conv2d(6,64,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
        self.conv_gcn1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv_gcn2 = torch.nn.Conv2d(64, 128, 1)

        self.conv_gcn2 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv_gcn3 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv_gcn4 = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv_gcn5 = nn.Sequential(nn.Conv2d(512 * 2, 1024, kernel_size=1, bias=False),
                                       self.bn6,
                                       nn.LeakyReLU(negative_slope=0.2))

        self.maxpool_gcn_6 = torch.nn.MaxPool2d((1, self.k), 1)  # 1,K
        self.maxpool_gcn_64 = torch.nn.MaxPool2d((1, self.k), 1)  # 1,K
        self.maxpool_gcn_128 = torch.nn.MaxPool2d((1, self.k), 1)  # 1,K
        self.maxpool_gcn_256 = torch.nn.MaxPool2d((1, self.k), 1)  # 1,K
        self.maxpool_gcn_512 = torch.nn.MaxPool2d((1, self.k), 1)  # 1,K
        self.maxpool_gcn_1024 = torch.nn.MaxPool2d((1, self.k), 1)#1,K
        self.maxpool_gcn_1920 = torch.nn.MaxPool2d((self.point_scales, 1), 1)  # 1,K





    def forward(self, x):
        #gcn
        #print(x.shape)
        x=x.transpose(2,1)
        x= get_graph_feature(x)
        x_1=self.conv_gcn1(x)
        x_1 = x_1.max(dim=-1, keepdim=False)[0]

        x_2 = get_graph_feature(x_1)
        x_2=self.conv_gcn2(x_2)
        x_2 = x_2.max(dim=-1, keepdim=False)[0]

        x_3 = get_graph_feature(x_2)
        x_3=self.conv_gcn3(x_3)
        x_3 = x_3.max(dim=-1, keepdim=False)[0]

        x_4 = get_graph_feature(x_3)
        x_4=self.conv_gcn4(x_4)
        x_4 = x_4.max(dim=-1, keepdim=False)[0]

        x_5 = get_graph_feature(x_4)
        x_5 = self.conv_gcn5(x_5)
        x_5 = x_5.max(dim=-1, keepdim=False)[0]
        x_2=x_2.unsqueeze(2)
        x_2=self.maxpool_gcn_128(x_2)
        x_2=x_2.squeeze(2)

        x_3 = x_3.unsqueeze(2)
        x_3 = self.maxpool_gcn_256(x_3)
        x_3 = x_3.squeeze(2)

        x_4 = x_4.unsqueeze(2)
        x_4 = self.maxpool_gcn_512(x_4)
        x_4 = x_4.squeeze(2)

        x_5 = x_5.unsqueeze(2)
        x_5 = self.maxpool_gcn_1024(x_5)
        x_5 = x_5.squeeze(2)


        L = [x_5, x_4, x_3, x_2]
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



