import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from operator import truediv
import warnings
import time
from torchsummary import summary
import math

warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline
import os
import matplotlib as mpl
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------- SWL Module ----------------- #
class SWL(nn.Module):
    def __init__(self, num_channels=256):
        super(SWL, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels // 8, out_channels=num_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x_avg, x_max = self.avgpool(x), self.maxpool(x)
        avg_weights = self.channel_attention(x_avg)
        max_weights = self.channel_attention(x_max)
        band_weights = self.sigmoid(avg_weights + max_weights)
        return band_weights


# ------------------- DEConv Module ------------------- #
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias
        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
        return res


# ------------------- Original ------------------- #
def bsm(n, d):
    a = [[0] * n for _ in range(n)]
    p, q = 0, n - 1
    w = (n + 1) // 2
    t = 0
    while p < d:
        for i in range(p, q):
            a[p][i] = t
        for i in range(p, q):
            a[i][q] = t
        for i in range(q, p, -1):
            a[q][i] = t
        for i in range(q, p, -1):
            a[i][p] = t
        p += 1
        q -= 1
    while p >= d and p < q:
        for i in range(p, q):
            a[p][i] = 1
        for i in range(p, q):
            a[i][q] = 1
        for i in range(q, p, -1):
            a[q][i] = 1
        for i in range(q, p, -1):
            a[i][p] = 1
        a[w - 1][w - 1] = 1
        p += 1
        q -= 1
    return np.array(a)


class ScaleMaskModule(nn.Module):
    def __init__(self, d):
        super(ScaleMaskModule, self).__init__()
        self.d = d

    def forward(self, x):
        w = x.shape[3]
        n = x.shape[2]
        o = x.shape[1]
        p = x.shape[0]
        out = bsm(w, self.d)
        out = torch.from_numpy(out).repeat(p, o, 1, 1).type(torch.FloatTensor).to(device)
        return x * out


class NCAM2D(nn.Module):
    def __init__(self, c, patch_size):
        super(NCAM2D, self).__init__()
        gamma, b = 2, 3
        kernel_size_21 = int(abs((math.log(c, 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size - 1) // 2 - 1)
        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2))
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2))

    def forward(self, x):
        out = x
        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x).reshape(out_x.shape[0], -1)
        out_x2 = reversed(out_x1.permute(1, 0)).permute(1, 0)
        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])
        out_xx = torch.cat([out_x1, out_x2], dim=2)

        out1 = self.avg_pool(out).reshape(out.shape[0], -1)
        out2 = reversed(out1.permute(1, 0)).permute(1, 0)
        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])
        outx = torch.cat([out1, out2], dim=2)

        at1 = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * F.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)
        at = F.sigmoid((at1 - 0.2) * 2)
        return out * at


class LE_DSC2D(nn.Module):
    def __init__(self, nin, nout, kernel_size_h, kernel_size_w, patch_size, padding=True):
        super(LE_DSC2D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM2D(self.nin, patch_size)
        self.at2 = NCAM2D(self.nout, patch_size)
        if padding:
            self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h, 1), padding=((kernel_size_h - 1) // 2, 0),
                                       groups=nin)
            self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1, kernel_size_w), padding=(0, (kernel_size_w - 1) // 2),
                                        groups=nin)
        else:
            self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h, 1), groups=nin)
            self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1, kernel_size_w), groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.depthwise(x) + self.depthwise1(x)
        out = self.at1(out)
        out = self.pointwise(out)
        out = self.at2(out)
        return out


class hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class LE_HCL_7x7(nn.Module):
    def __init__(self, ax, aa, c, pca_components, patch_size):
        super(LE_HCL_7x7, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, ax, kernel_size=(c, c, c), padding=((c - 1) // 2, (c - 1) // 2, (c - 1) // 2)),
            nn.BatchNorm3d(ax),
            hswish()
        )
        # 修复：确保输出通道数为 pca_components
        self.conv2d = nn.Sequential(
            LE_DSC2D(aa, pca_components, c, c, patch_size),  # 改为输出 pca_components
            nn.BatchNorm2d(pca_components),
            hswish()
        )

    def forward(self, x):
        out = self.conv3d(x)
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.conv2d(out)
        # 确保残差连接维度匹配
        return out + x.squeeze(1)  # 修改这里，将5D的x转换为4D


# ------------------- Main Model: Lite_HCNet_SWL_DE ------------------- #
class CAB_DNet(nn.Module):
    def __init__(self, in_channels, class_num, patch_size, num_bands):
        super(CAB_DNet, self).__init__()

        # 保存参数
        self.patch_size = patch_size
        self.in_channels = in_channels

        # SWL Module for spectral weight learning
        self.swl = SWL(num_channels=num_bands)

        # DEConv branch (replace 3x3 LE-HCL)
        self.deconv_branch = nn.Sequential(
            DEConv(dim=in_channels),
            nn.BatchNorm2d(in_channels),
            hswish()
        )

        # 7x7 LE-HCL branch
        e = 3
        self.le_hcl_branch = LE_HCL_7x7(e, e * in_channels, 7, in_channels, patch_size)

        # Classification head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, class_num)

    def forward(self, x, return_band_weights=False, center_weights=None, labels=None):
        """
        适配data_loader.py的输出格式和main_train.py的处理
        """
        # 检查输入维度并适配
        if x.dim() == 5:
            # 如果输入是5D [B, 1, C, H, W]，转换为4D [B, C, H, W]
            x = x.squeeze(1)
        elif x.dim() == 4:
            # 如果已经是4D [B, C, H, W]，直接使用
            pass
        else:
            # 其他情况，尝试reshape
            x = x.view(-1, self.in_channels, self.patch_size, self.patch_size)

        # x: [B, C, H, W]
        band_weights = self.swl(x)
        x = x + x * band_weights

        # DEConv branch
        out_de = self.deconv_branch(x)

        # 7x7 LE-HCL branch (need 5D input)
        x_5d = x.unsqueeze(1)  # [B, 1, C, H, W]
        out_le = self.le_hcl_branch(x_5d)

        # Feature fusion - 确保维度匹配
        out = out_de + out_le

        # Classification
        out = self.avg_pool(out)
        out = out.reshape(out.shape[0], -1)
        logits = self.fc1(out)

        if return_band_weights and center_weights is not None and labels is not None:
            w_flat = band_weights.view(band_weights.size(0), -1)
            center_batch = center_weights[labels]
            lc_loss = F.mse_loss(w_flat, center_batch, reduction='mean')
            return logits, band_weights, lc_loss
        elif return_band_weights:
            return logits, band_weights
        else:
            return logits


if __name__ == '__main__':

    batch_size = 16
    pca_components = 15  # 示例值
    patch_size = 9  # 示例值
    num_classes = 10  # 示例值

    model = Lite_HCNet_SWL_DE(
        in_channels=pca_components,
        class_num=num_classes,
        patch_size=patch_size,
        num_bands=pca_components
    ).cuda()

    # 模拟data_loader输出的原始数据 [B, 1, C, H, W]
    x_5d = torch.randn(batch_size, 1, pca_components, patch_size, patch_size).cuda()
    # 模拟main_train.py中的view操作
    x_4d = x_5d.view(-1, pca_components, patch_size, patch_size)

    y = model(x_4d)
    print("Output shape:", y.shape)  # [16, 10]
    print("Model is ready for training with main_train.py")

    # 测试维度是否正确
    assert y.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {y.shape}"
    print("Dimension test passed!")