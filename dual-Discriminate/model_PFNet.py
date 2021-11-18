#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

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
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        # self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.ca = ChannelAttention(1920)
        self.ca1 = ChannelAttention(1024)
        self.ca2 = ChannelAttention(512)
        self.ca3 = ChannelAttention(256)
        self.ca4 = ChannelAttention(128)
        self.sa = SpatialAttention()
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        # print(x.shape)
        # 三次下采样,提特征
        # input  24,2048,3
        x = torch.unsqueeze(x, 1)
        # 24,1,2048,3
        x = F.relu(self.bn1(self.conv1(x)))
        # 24,64,2048,1
        # x = F.relu(self.bn2(self.conv2(x)))

        x_128 = F.relu(self.bn3(self.conv3(x)))
        # 24,128,2048,1
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        # 24,256,2048,1
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        # 24,1024,2048,1
        # 注意力机制
        # x_1024 = self.ca1(x_1024) * x_1024
        # x_1024=self.sa(x_1024)*x_1024
        # x_512 = self.ca2(x_512) * x_512
        # x_512=self.sa(x_512)*x_512
        # x_256 = self.ca3(x_256) * x_256
        # x_256 = self.sa(x_256) * x_256
        # x_128 = self.ca4(x_128) * x_128
        # x_128 = self.sa(x_128) * x_128
        # 最大池化
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        # print(x_128.shape)
        # 24,128,1
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        # 24,1024,1
        L = [x_1024, x_512, x_256, x_128]
        # 24,1920,1
        x = torch.cat(L, 1)
        # x = x.unsqueeze(2)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        # x = x.squeeze(3)
        # print(x.shape)
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

    def forward(self, x):
        # print(x.shape)
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
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
        return latentfeature


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
        # print(crop_point_num)
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, self.crop_point_num*2)
        self.fc2 = nn.Linear(self.crop_point_num*2, int(self.crop_point_num))
        self.fc3 = nn.Linear( int(self.crop_point_num),  int(self.crop_point_num/2))

        self.fc1_1 = nn.Linear(self.crop_point_num*2, int(self.crop_point_num/4) * self.crop_point_num)
        self.fc2_1 = nn.Linear(int(self.crop_point_num), int(self.crop_point_num/8) *  int(self.crop_point_num/4))  # nn.Linear(2048,1024*2048) !
        self.fc3_1 = nn.Linear( int(self.crop_point_num/2), int(self.crop_point_num/8) * 3)

        #        self.bn1 = nn.BatchNorm1d(1024)
        #        self.bn2 = nn.BatchNorm1d(512)
        #        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
        #        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
        #        self.bn5 = nn.BatchNorm1d(64*128)
        #
        self.conv1_1 = torch.nn.Conv1d(self.crop_point_num, self.crop_point_num, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(self.crop_point_num, int(self.crop_point_num/2), 1)
        self.conv1_3 = torch.nn.Conv1d(int(self.crop_point_num/2),12, 1)
        self.conv2_1 = torch.nn.Conv1d(int(self.crop_point_num/4), 6, 1)  # torch.nn.Conv1d(256,12,1) !

    #        self.bn1_ = nn.BatchNorm1d(512)
    #        self.bn2_ = nn.BatchNorm1d(256)

    def forward(self, x):
        # print(x.shape)
        x = self.latentfeature(x)
        # print(x.shape)
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256
        # print(x_3.shape)
        pc1_feat = self.fc3_1(x_3)
        # print(pc1_feat.shape)
        pc1_xyz = pc1_feat.reshape(-1, int(self.crop_point_num/8), 3)  # 512x3 center1

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, int(self.crop_point_num/4), int(self.crop_point_num/8))
        # print(pc2_feat.shape)
        pc2_xyz = self.conv2_1(pc2_feat)  # 6x64 center2

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, self.crop_point_num, int(self.crop_point_num/4))
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        # print(pc3_feat.shape)
        pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, int(self.crop_point_num/8), 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, int(self.crop_point_num/4), 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, int(self.crop_point_num/4), int(self.crop_point_num / 128), 3)
        # print(pc3_xyz.shape)
        # print(pc2_xyz_expand.shape)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)

        return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,fine

class _netG_single(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
        super(_netG_single, self).__init__()
        self.crop_point_num = crop_point_num
        # print(crop_point_num)
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear( 512,  256)
        self.fc4=nn.Linear(1792,crop_point_num*3)
        # self.fc1_1 = nn.Linear(self.crop_point_num*2, int(self.crop_point_num/2) * self.crop_point_num)
        # self.fc2_1 = nn.Linear(int(self.crop_point_num/2), int(self.crop_point_num/4) *  int(self.crop_point_num/2))  # nn.Linear(2048,1024*2048) !
        # self.fc3_1 = nn.Linear( int(self.crop_point_num/4), int(self.crop_point_num/4) * 3)

        #        self.bn1 = nn.BatchNorm1d(1024)
        #        self.bn2 = nn.BatchNorm1d(512)
        #        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
        #        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
        #        self.bn5 = nn.BatchNorm1d(64*128)
        #
        # self.conv1_1 = torch.nn.Conv1d(self.crop_point_num, self.crop_point_num, 1)  # torch.nn.Conv1d(256,256,1) !
        # self.conv1_2 = torch.nn.Conv1d(self.crop_point_num, int(self.crop_point_num/2), 1)
        # self.conv1_3 = torch.nn.Conv1d(int(self.crop_point_num/2), 6, 1)
        # self.conv2_1 = torch.nn.Conv1d(int(self.crop_point_num/2), 6, 1)  # torch.nn.Conv1d(256,12,1) !

    #        self.bn1_ = nn.BatchNorm1d(512)
    #        self.bn2_ = nn.BatchNorm1d(256)

    def forward(self, x):
        # print(self.crop_point_num)
        x = self.latentfeature(x)
        # print(x.shape)
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        x_4=[x_1,x_2,x_3]
        x_4=torch.cat(x_4,1)
        # print(x_4.shape)
        x_4=F.relu(self.fc4(x_4))
        x_4 = x_4.reshape(-1, self.crop_point_num, 3)

        return x_4


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
        # print(x.shape)
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
class _netlocalD_deepimage(nn.Module):
    def __init__(self,M_size,N_size):
        super(_netlocalD_deepimage, self).__init__()
        self.M_size=M_size
        self.N_size=N_size
        self.conv1 = torch.nn.Conv2d(self.M_size, self.M_size, (5, 5),stride=1)
        self.conv2 = torch.nn.Conv2d(self.M_size, self.M_size, (5, 5),stride=1)

        self.conv3 = torch.nn.Conv2d(self.M_size, self.M_size, (5, 5))

        self.maxpool1 = torch.nn.MaxPool2d((5,5), 1)
        self.maxpool2 = torch.nn.MaxPool2d((5,5), 1)

        self.bn1 = nn.BatchNorm2d(self.M_size)
        self.bn2 = nn.BatchNorm2d(self.M_size)

        self.bn3 = nn.BatchNorm2d(self.M_size)
        self.bn4 = nn.BatchNorm2d(self.M_size)
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 1)
    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x=self.maxpool1(x)
        print(x.shape)
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.maxpool2(x)
        # print(x.shape)
        x=torch.reshape(x,[x.shape[0],x.shape[1],1,-1])
        # x=torch.squeeze(x,2)
        print(x.shape)
        x=F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        x=torch.squeeze(x,2)
        x = torch.squeeze(x, 2)
        # print(x.shape)
        return x

if __name__ == '__main__':
    input1 = torch.randn(64, 2048, 3)
    input2 = torch.randn(64, 512, 3)
    input3 = torch.randn(64, 256, 3)

    image_input=torch.randn(32,1,64,64)
    D_input2 = torch.randn(64, 1,512, 3)
    input_ = [input1, input2, input3]
    netG = _netG(3, 1, [2048, 512, 256], 1024)
    # output = netG(input_)
    netD=_netlocalD_deepimage(image_input.shape[1],image_input.shape[2])
    output=netD(image_input)

    netD_D=_netlocalD(512)
    output=netD_D(D_input2)

    # print(output)
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data
#
#
# # 通道注意力
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# # 空间注意力
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class Convlayer(nn.Module):
#     def __init__(self, point_scales):
#         super(Convlayer, self).__init__()
#         self.point_scales = point_scales
#         self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
#         self.conv2 = torch.nn.Conv2d(64, 64, 1)
#         self.conv3 = torch.nn.Conv2d(64, 128, 1)
#         self.conv4 = torch.nn.Conv2d(128, 256, 1)
#         self.conv5 = torch.nn.Conv2d(256, 512, 1)
#         self.conv6 = torch.nn.Conv2d(512, 1024, 1)
#         self.ca = ChannelAttention(1920)
#         self.ca1 = ChannelAttention(1024)
#         self.ca2 = ChannelAttention(512)
#         self.ca3 = ChannelAttention(256)
#         self.ca4 = ChannelAttention(128)
#         self.sa = SpatialAttention()
#         self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.bn6 = nn.BatchNorm2d(1024)
#
#     def forward(self, x):
#         # 三次下采样,提特征
#         # input  24,2048,3
#         x = torch.unsqueeze(x, 1)
#         # 24,1,2048,3
#         x = F.relu(self.bn1(self.conv1(x)))
#         # 24,64,2048,1
#         x = F.relu(self.bn2(self.conv2(x)))
#         x_128 = F.relu(self.bn3(self.conv3(x)))
#         # 24,128,2048,1
#         x_256 = F.relu(self.bn4(self.conv4(x_128)))
#         # 24,256,2048,1
#         x_512 = F.relu(self.bn5(self.conv5(x_256)))
#         x_1024 = F.relu(self.bn6(self.conv6(x_512)))
#         # 24,1024,2048,1
#         # 注意力机制
#         # x_1024 = self.ca1(x_1024) * x_1024
#         # x_1024=self.sa(x_1024)*x_1024
#         # x_512 = self.ca2(x_512) * x_512
#         # x_512=self.sa(x_512)*x_512
#         # x_256 = self.ca3(x_256) * x_256
#         # x_256 = self.sa(x_256) * x_256
#         # x_128 = self.ca4(x_128) * x_128
#         # x_128 = self.sa(x_128) * x_128
#         # 最大池化
#         L = [x_1024, x_512, x_256, x_128]
#         # 24,1920,1
#         x = torch.cat(L, 1)
#         x = self.ca(x) * x
#         x = self.sa(x) * x
#         x=torch.squeeze(self.maxpool(x), 2)
#         # # 最大池化
#         # x_128 = torch.squeeze(self.maxpool(x_128), 2)
#         # # 24,128,1
#         # x_256 = torch.squeeze(self.maxpool(x_256), 2)
#         # x_512 = torch.squeeze(self.maxpool(x_512), 2)
#         # x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
#         # # 24,1024,1
#         # L = [x_1024, x_512, x_256, x_128]
#         #
#         # # 24,1920,1
#         # x = torch.cat(L, 1)
#         # x = x.unsqueeze(2)
#         # x = self.ca(x) * x
#         # x = self.sa(x) * x
#         # x = x.squeeze(3)
#
#
#         return x
#
#
# class Latentfeature(nn.Module):
#     def __init__(self, num_scales, each_scales_size, point_scales_list):
#         super(Latentfeature, self).__init__()
#         self.num_scales = num_scales
#         self.each_scales_size = each_scales_size
#         self.point_scales_list = point_scales_list
#         self.Convlayers1 = nn.ModuleList(
#             [Convlayer(point_scales=self.point_scales_list[0]) for i in range(self.each_scales_size)])
#         self.Convlayers2 = nn.ModuleList(
#             [Convlayer(point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
#         self.Convlayers3 = nn.ModuleList(
#             [Convlayer(point_scales=self.point_scales_list[2]) for i in range(self.each_scales_size)])
#         self.conv1 = torch.nn.Conv1d(3, 1, 1)
#         self.bn1 = nn.BatchNorm1d(1)
#
#     def forward(self, x):
#         outs = []
#         for i in range(self.each_scales_size):
#             outs.append(self.Convlayers1[i](x[0]))
#         for j in range(self.each_scales_size):
#             outs.append(self.Convlayers2[j](x[1]))
#         for k in range(self.each_scales_size):
#             outs.append(self.Convlayers3[k](x[2]))
#         latentfeature = torch.cat(outs, 2)
#         latentfeature = latentfeature.transpose(1, 2)
#         latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
#
#         latentfeature = torch.squeeze(latentfeature, 1)
#         #        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))
#         #        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
#         #        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
#         #        latentfeature = latentfeature + latentfeature_64
#         #        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
#         #        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
#         #        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
#         #        latentfeature = latentfeature + latentfeature_256
#         #        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
#         #        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
#         #        latentfeature = self.maxpool(latentfeature)
#         #        latentfeature = torch.squeeze(latentfeature,2)
#         return latentfeature
#
#
# class PointcloudCls(nn.Module):
#     def __init__(self, num_scales, each_scales_size, point_scales_list, k=40):
#         super(PointcloudCls, self).__init__()
#         self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
#         self.fc1 = nn.Linear(1920, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.latentfeature(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = F.relu(self.bn3(self.dropout(self.fc3(x))))
#         x = self.fc4(x)
#         return F.log_softmax(x, dim=1)
#
#
# class _netG(nn.Module):
#     def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
#         super(_netG, self).__init__()
#         self.crop_point_num = crop_point_num
#         self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
#         self.fc1 = nn.Linear(1920, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#
#         self.fc1_1 = nn.Linear(1024, 128 * 512)
#         self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
#         self.fc3_1 = nn.Linear(256, 64 * 3)
#
#         #        self.bn1 = nn.BatchNorm1d(1024)
#         #        self.bn2 = nn.BatchNorm1d(512)
#         #        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#         #        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#         #        self.bn5 = nn.BatchNorm1d(64*128)
#         #
#         self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
#         self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
#         self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !
#
#     #        self.bn1_ = nn.BatchNorm1d(512)
#     #        self.bn2_ = nn.BatchNorm1d(256)
#
#     def forward(self, x):
#         x = self.latentfeature(x)
#         x_1 = F.relu(self.fc1(x))  # 1024
#         x_2 = F.relu(self.fc2(x_1))  # 512
#         x_3 = F.relu(self.fc3(x_2))  # 256
#
#         pc1_feat = self.fc3_1(x_3)
#         pc1_xyz = pc1_feat.reshape(-1, 64, 3)  # 64x3 center1
#
#         pc2_feat = F.relu(self.fc2_1(x_2))
#         pc2_feat = pc2_feat.reshape(-1, 128, 64)
#         pc2_xyz = self.conv2_1(pc2_feat)  # 6x64 center2
#
#         pc3_feat = F.relu(self.fc1_1(x_1))
#         pc3_feat = pc3_feat.reshape(-1, 512, 128)
#         pc3_feat = F.relu(self.conv1_1(pc3_feat))
#         pc3_feat = F.relu(self.conv1_2(pc3_feat))
#         pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine
#
#         pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
#         pc2_xyz = pc2_xyz.transpose(1, 2)
#         pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)
#         pc2_xyz = pc1_xyz_expand + pc2_xyz
#         pc2_xyz = pc2_xyz.reshape(-1, 128, 3)
#
#         pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
#         pc3_xyz = pc3_xyz.transpose(1, 2)
#         pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.crop_point_num / 128), 3)
#         pc3_xyz = pc2_xyz_expand + pc3_xyz
#         pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)
#
#         return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,fine
#
#
# class _netlocalD(nn.Module):
#     def __init__(self, crop_point_num):
#         super(_netlocalD, self).__init__()
#         self.crop_point_num = crop_point_num
#         self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
#         self.conv2 = torch.nn.Conv2d(64, 64, 1)
#         self.conv3 = torch.nn.Conv2d(64, 128, 1)
#         self.conv4 = torch.nn.Conv2d(128, 256, 1)
#         self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
#         self.avgpool = torch.nn.AvgPool2d((self.crop_point_num, 1), 1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(448, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 16)
#         self.fc4 = nn.Linear(16, 1)
#         self.bn_1 = nn.BatchNorm1d(256)
#         self.bn_2 = nn.BatchNorm1d(128)
#         self.bn_3 = nn.BatchNorm1d(16)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x_64 = F.relu(self.bn2(self.conv2(x)))
#         x_128 = F.relu(self.bn3(self.conv3(x_64)))
#         x_256 = F.relu(self.bn4(self.conv4(x_128)))
#         x_64 = self.maxpool(x_64)
#         x_128 = self.maxpool(x_128)
#         x_256 = self.maxpool(x_256)
#         x_64 = torch.squeeze(torch.squeeze(x_64, -1), -1)
#         x_128 = torch.squeeze(torch.squeeze(x_128, -1), -1)
#         x_256 = torch.squeeze(torch.squeeze(x_256, -1), -1)
#
#         Layers = [x_256, x_128, x_64]
#         x = torch.cat(Layers, 1)
#
#         x = F.relu(self.bn_1(self.fc1(x)))
#         x = F.relu(self.bn_2(self.fc2(x)))
#         x = F.relu(self.bn_3(self.fc3(x)))
#         x = self.fc4(x)
#         return x
#
#
# if __name__ == '__main__':
#     input1 = torch.randn(64, 2048, 3)
#     input2 = torch.randn(64, 512, 3)
#     input3 = torch.randn(64, 256, 3)
#     input_ = [input1, input2, input3]
#     netG = _netG(3, 1, [2048, 512, 256], 1024)
#     output = netG(input_)
#     print(output)
