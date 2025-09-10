# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torch.distributions import Bernoulli
from ELA import * ##

############################################## 
#             Basic ResNet model             #
##############################################

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:

            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

# Bottleneck block【瓶颈残差块】-【先将数据集中到窄通道（瓶颈）高效处理，再分发】-【ResNet50及以上需要】
class BottleneckBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4) # 计算瓶颈层的输出维度，为输出维度 outdim 的四分之一。
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)  # 压缩通道【将输入特征图的通道数从 indim 减少到 bottleneckdim】
        self.BN1 = nn.BatchNorm2d(bottleneckdim) # 批量归一化
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1, padding=1)  # 空间特征提取【进一步减少特征图的空间维度】
        self.BN2 = nn.BatchNorm2d(bottleneckdim) # 批量归一化
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)  # 扩展通道【将通道数从 bottleneckdim 增加到 outdim】
        self.BN3 = nn.BatchNorm2d(outdim) # 批量归一化
        #self.ela= ELA(in_channels=outdim, phi='T')
        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        ##out = self.ela(out) ###
        out = out + short_out #残差连接
        #out = self.ela(out)
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    maml = False  # Default

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0) and i != 3
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            # self.final_feat_dim = indim

        self.feat_dim = [512, 14, 14]
        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        print(input.size())  # 或者 input.shape
        out = self.trunk(x)
        # out = out.view(out.size(0), -1)
        return out


def ResNet10(flatten=True):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten)


def ResNet18(flatten=False):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten)


def ResNet34(flatten=True):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten)


def ResNet50(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], [256, 512, 1024, 2048], flatten)


def ResNet101(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], [256, 512, 1024, 2048], flatten)


############################################## 
#        a variant of ResNet model           #
##############################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        #print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
        block_mask = 1 - padded_mask#[:height, :width]
        return block_mask
    

class BaseBlockNet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        super(BaseBlockNet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes) # 卷积层
        self.bn1 = nn.BatchNorm2d(planes) # 批量归一化层
        self.relu = nn.LeakyReLU(0.1) # LeakyReLU激活函数，用于引入非线性
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride) # 最大池化
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se: # 是否使用SE（Squeeze-and-Excitation，即注意力机制）层
            self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class LAnet(nn.Module):
    def __init__(self, in_channels, phi):
        super(LAnet, self).__init__()
        """
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        
        参数:
        - in_channels (int): 输入特征图的通道数
        - phi (str): 表示卷积核大小和组数的选择，'T', 'B', 'S', 'L'中的一个
        """
        # 根据 phi 参数选择不同的卷积核大小
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        # 根据 phi 参数选择不同的卷积组数
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        # 根据 phi 参数选择不同的归一化组数
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi] 
        # 计算填充大小以保持卷积后尺寸不变
        pad = Kernel_size // 2
        # 1D 卷积层，使用分组卷积，卷积核大小为 Kernel_size
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        # 组归一化层
        self.GN = nn.GroupNorm(num_groups, in_channels)
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input):
        """
        前向传播函数。
        参数:
        - input (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
        返回:
        - torch.Tensor: 应用边缘注意力后的特征图
        """
        b, c, h, w = input.size()  # 获取输入特征图的形状
        # 在宽度方向上进行平均池化
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        # 在高度方向上进行平均池化
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        # 对池化后的特征图应用 1D 卷积
        x_h = self.con1(x_h)  # [b, c, h]
        x_w = self.con1(x_w)  # [b, c, w]
        # 对卷积后的特征图进行归一化和激活，并 reshape 回来
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        # 将输入特征图、x_h 和 x_w 按元素相乘，得到最终的输出特征图
        return x_h * x_w * input

class resnet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(resnet, self).__init__()

        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=1, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.feat_dim = [640, 10, 10]
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)
        
        self.ela = LAnet(in_channels=640, phi='T') ###ELA边缘注意力机制

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ela(x) ###
        return x

def ResNet12(keep_prob=1.0, avg_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = resnet(BaseBlockNet, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model

def ResNet34s(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-24 model.
    """
    model = resnet(BaseBlockNet, [2, 3, 4, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


