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
from osgeo import gdal


def trans_tif(image, output_path):
    if len(image.shape) == 3:
        bands = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]
    else:
        bands = 1
        height = image.shape[0]
        width = image.shape[1]
    if 'uint8' in image.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'uint16' in image.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'float64' in image.dtype.name:
        datatype = gdal.GDT_Float64
    else:
        datatype = gdal.GDT_Float32
    # 创建文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, bands, datatype)

    if len(image.shape) == 3:
        for i in range(image.shape[0]):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(image[i])  # 写入第 i+1 个波段
            band.SetNoDataValue(0)  # 设置 NoData 值 (可选
    else:
        band = dataset.GetRasterBand(1).WriteArray(image)


# PCA降维
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # 获得 y 中的标记样本数
    # 实图像中大部分为是0，没有label，我们要取的，只是有label的部分。
    # 因此，先做个循环，看看有多少个像素有 label，然后记录在 count 里，这样 count 比以前的X.shape[0] * X.shape[1]要小很多，内存就不会爆了。
    count = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r, c] != 0:
                count = count + 1

    # split patches
    patchesData = np.zeros((count, windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((count))
    count = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] != 0:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[count, :, :, :] = patch
                patchesLabels[count] = y[r - margin, c - margin]
                count = count + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        # 必须减1，因为原始标签从1开始，但模型需要0-based标签
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


# 数据集设置

# IP数据集
# data = 'IP'
# class_num = 16
# X = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\Indian_pines_corrected.mat')['indian_pines_corrected']
# y = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\Indian_pines_gt.mat')['indian_pines_gt']
# patch_size =19 # 每个像素周围提取 patch 的尺寸
# pca_components =15 # 使用 PCA 降维，得到主成分的数量

# SA数据集
# data = 'SA'
# class_num = 16
# X = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\Salinas_corrected.mat')['salinas_corrected']
# y = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\Salinas_gt.mat')['salinas_gt']
# HU数据集
# data = 'HU'
# class_num = 15
# X = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\HoustonU\Houston.mat')['Houston']
# y = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\HoustonU\Houston_gt.mat')['Houston_gt']
# patch_size =7 # 每个像素周围提取 patch 的尺寸
# pca_components =17 # 使用 PCA 降维，得到主成分的数量
# YRD数据集
# data = 'YRD'
# class_num = 8
# X = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\GF5_462_617.mat')['data']
# y = sio.loadmat('D:\PycharmProjects\HSI\Various documents\data\gt_GF5_462_617.mat')['label']
# patch_size =11 # 每个像素周围提取 patch 的尺寸
# pca_components =17 # 使用 PCA 降维，得到主成分的数量

class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    # 随机生成长度为len(a)的序列，将a和b按同样的随机顺序重新排列
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data(pixels, labels, percent, splitdset="custom", rand_state=345):
    """
    :param pixels: len(pixels.shape) >3表示cube，小于则表示location
    :param labels: 标签
    :param percent: 训练集的比重，为整数时，表示每一类选取多少个作为训练集（splitdset="custom"时）
    :param mode: CNN模式划分训练集和测试集，GAN模式只需要训练集
    :param splitdset: 使用sklearn还是自己设计的划分方式，“sklearn”表示用sklearn，“custom”表示自己的
    :param rand_state: 保证每次的划分方式相同
    :return:
    """
    if splitdset == "sklearn":
        # train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
        return train_test_split(pixels, labels, test_size=percent, stratify=labels, random_state=rand_state)
    elif splitdset == "custom":

        # np.unique:该函数是去除数组中的重复数字，并进行排序之后输出。return_counts = true,返回个数（用于统计各个元素出现的次数）
        # pixels_number:各类标签的个数，按顺序记录在列表中。
        pixels_number = np.unique(labels, return_counts=True)[1]
        # 设置各类地物训练集的大小
        # train_set_size = np.ones(len(pixels_number)) * percent

        # train_set_size = [1, 14, 8, 2, 5, 7, 1, 5, 1, 10, 25, 6, 2, 13, 4, 1]
        train_set_size = [66, 186, 20, 30, 13, 50, 13, 36, 9]
        # 训练集总大小
        tr_size = int(sum(train_set_size))
        # 测试集总大小
        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
        sizetr = np.array([tr_size] + list(pixels.shape[1:]))
        sizete = np.array([te_size] + list(pixels.shape[1:]))
        train_x = np.empty((sizetr));
        train_y = np.empty((tr_size))
        test_x = np.empty((sizete));
        test_y = np.empty((te_size))
        trcont = 0
        tecont = 0
        for cl in np.unique(labels):
            pixels_cl = pixels[labels == cl]
            labels_cl = labels[labels == cl]
            pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
            for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):

                if cont < train_set_size[int(cl)]:
                    train_x[trcont, :, :, :] = a
                    train_y[trcont] = b
                    trcont += 1
                else:
                    test_x[tecont, :, :, :] = a
                    test_y[tecont] = b
                    tecont += 1
        train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
        test_x, test_y = random_unison(test_x, test_y, rstate=rand_state)
        return train_x, test_x, train_y, test_y


def build_data_loader(args):
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9
        args.patch_size = 7
        args.PCA = 12
    if args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 15
        args.patch_size = 7
        args.PCA = 18
    if args.dataset == 'IP':
        X = sio.loadmat('./oridata/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian/Indian_pines_gt.mat')['indian_pines_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 19
        args.PCA = 16
    if args.dataset == 'salinas':
        X = sio.loadmat('./oridata/salinas/salinas.mat')['salinas']
        y = sio.loadmat('./oridata/salinas/salinas_gt.mat')['salinas_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 19
        args.PCA = 14
        # args.PCA = 30（保留约99.9 % 方差）
        # args.PCA = 15（保留约99 % 方差）
        # args.PCA = 10（保留约97 % 方差）
        # args.PCA = 0.99（自动确定达到99 % 方差的组件数）
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    if args.is_train:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')

            X_pca, y = createImageCubes(X_pca, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X_pca.shape)
            print('Data cube y shape: ', y.shape)

            print('\n... ... create train & test data ... ...')
            Xtrain, Xtest, ytrain, ytest = split_data(X_pca, y, 1 - args.train_ratio, splitdset="sklearn")
            print('Xtrain shape: ', Xtrain.shape)
            print('Xtest  shape: ', Xtest.shape)

            # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
            Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
            Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
            print('before transpose: Xtrain shape: ', Xtrain.shape)
            print('before transpose: Xtest  shape: ', Xtest.shape)
        else:
            print('\n... ... create data cubes ... ...')
            X, y = createImageCubes(X, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X.shape)
            print('Data cube y shape: ', y.shape)

            print('\n... ... create train & test data ... ...')
            Xtrain, Xtest, ytrain, ytest = split_data(X, y, 1 - args.train_ratio, splitdset="sklearn")
            print('Xtrain shape: ', Xtrain.shape)
            print('Xtest  shape: ', Xtest.shape)

            # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
            Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.hsi_bands, 1)
            Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.hsi_bands, 1)
            print('before transpose: Xtrain shape: ', Xtrain.shape)
            print('before transpose: Xtest  shape: ', Xtest.shape)

        # 为了适应 pytorch 结构，数据要做 transpose
        Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2)
        print('after transpose: Xtrain shape: ', Xtrain.shape)
        print('after transpose: Xtest  shape: ', Xtest.shape)
        # 创建 trainloader 和 testloader
        trainset = TrainDS(Xtrain, ytrain)
        testset = TestDS(Xtest, ytest)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=256, shuffle=False, num_workers=0)
        return train_loader, test_loader
    else:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')

            X = padWithZeros(X_pca, args.patch_size // 2)
        else:
            X = padWithZeros(X, args.patch_size // 2)
        return X, y


def build_data_sim_loader(args):  # 不改patch_size
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9
        args.patch_size = 9
        args.PCA = 12
    if args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 15
        args.patch_size = 9
        args.PCA = 18
    if args.dataset == 'IP':
        X = sio.loadmat('./oridata/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian/Indian_pines_gt.mat')['indian_pines_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 9
        args.PCA = 16
    if args.dataset == 'salinas':
        X = sio.loadmat('./oridata/salinas/salinas.mat')['salinas']
        y = sio.loadmat('./oridata/salinas/salinas_gt.mat')['salinas_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 9
        args.PCA = 14
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    if args.is_train:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')

            X_pca, y = createImageCubes(X_pca, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X_pca.shape)
            print('Data cube y shape: ', y.shape)

            print('\n... ... create train & test data ... ...')
            Xtrain, Xtest, ytrain, ytest = split_data(X_pca, y, 1 - args.train_ratio, splitdset="sklearn")
            print('Xtrain shape: ', Xtrain.shape)
            print('Xtest  shape: ', Xtest.shape)

            # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
            Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
            Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
            print('before transpose: Xtrain shape: ', Xtrain.shape)
            print('before transpose: Xtest  shape: ', Xtest.shape)
        else:
            print('\n... ... create data cubes ... ...')
            X, y = createImageCubes(X, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X.shape)
            print('Data cube y shape: ', y.shape)

            print('\n... ... create train & test data ... ...')
            Xtrain, Xtest, ytrain, ytest = split_data(X, y, 1 - args.train_ratio, splitdset="sklearn")
            print('Xtrain shape: ', Xtrain.shape)
            print('Xtest  shape: ', Xtest.shape)

            # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
            Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.hsi_bands, 1)
            Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.hsi_bands, 1)
            print('before transpose: Xtrain shape: ', Xtrain.shape)
            print('before transpose: Xtest  shape: ', Xtest.shape)

        # 为了适应 pytorch 结构，数据要做 transpose
        Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2)
        print('after transpose: Xtrain shape: ', Xtrain.shape)
        print('after transpose: Xtest  shape: ', Xtest.shape)
        # 创建 trainloader 和 testloader
        trainset = TrainDS(Xtrain, ytrain)
        testset = TestDS(Xtest, ytest)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=256, shuffle=False, num_workers=0)
        return train_loader, test_loader
    else:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')

            X = padWithZeros(X_pca, args.patch_size // 2)
        else:
            X = padWithZeros(X, args.patch_size // 2)
        return X, y


def build_data_cacf_loader(args):  # 原文不用PCA
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9
        args.patch_size = 7
    if args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 15
        args.patch_size = 7
    if args.dataset == 'IP':
        X = sio.loadmat('./oridata/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian/Indian_pines_gt.mat')['indian_pines_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 7
    if args.dataset == 'salinas':
        X = sio.loadmat('./oridata/salinas/salinas.mat')['salinas']
        y = sio.loadmat('./oridata/salinas/salinas_gt.mat')['salinas_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 7
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    if args.is_train:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')

            X_pca, y = createImageCubes(X_pca, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X_pca.shape)
            print('Data cube y shape: ', y.shape)

            print('\n... ... create train & test data ... ...')
            Xtrain, Xtest, ytrain, ytest = split_data(X_pca, y, 1 - args.train_ratio, splitdset="sklearn")
            print('Xtrain shape: ', Xtrain.shape)
            print('Xtest  shape: ', Xtest.shape)

            # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
            Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
            Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
            print('before transpose: Xtrain shape: ', Xtrain.shape)
            print('before transpose: Xtest  shape: ', Xtest.shape)
        else:
            print('\n... ... create data cubes ... ...')
            X, y = createImageCubes(X, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X.shape)
            print('Data cube y shape: ', y.shape)

            print('\n... ... create train & test data ... ...')
            Xtrain, Xtest, ytrain, ytest = split_data(X, y, 1 - args.train_ratio, splitdset="sklearn")
            print('Xtrain shape: ', Xtrain.shape)
            print('Xtest  shape: ', Xtest.shape)

            # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
            Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.hsi_bands, 1)
            Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.hsi_bands, 1)
            print('before transpose: Xtrain shape: ', Xtrain.shape)
            print('before transpose: Xtest  shape: ', Xtest.shape)

        # 为了适应 pytorch 结构，数据要做 transpose
        Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2)
        print('after transpose: Xtrain shape: ', Xtrain.shape)
        print('after transpose: Xtest  shape: ', Xtest.shape)
        # 创建 trainloader 和 testloader
        trainset = TrainDS(Xtrain, ytrain)
        testset = TestDS(Xtest, ytest)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=256, shuffle=False, num_workers=0)
        return train_loader, test_loader
    else:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')

            X = padWithZeros(X_pca, args.patch_size // 2)
        else:
            X = padWithZeros(X, args.patch_size // 2)
        return X, y


def build_data_d32_loader(args):
    """
    专门为D32模型设计的数据加载器
    根据论文设置：
    - SA和PU数据集: PCA=15, patch_size=15
    - IP数据集: PCA=30, patch_size=15
    - Houston数据集: PCA=18, patch_size=15
    """
    # 根据数据集设置PCA和patch_size
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9
        args.patch_size = 15  # 论文Fig. 14显示15×15是最佳空间尺寸
        args.PCA = 15  # 论文Fig. 15显示PU使用15个主成分时准确率最高
    elif args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 15
        args.patch_size = 15
        args.PCA = 18  # Houston通常使用18个主成分
    elif args.dataset == 'IP':
        X = sio.loadmat('./oridata/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian/Indian_pines_gt.mat')['indian_pines_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 15  # 论文Fig. 14显示15×15是最佳空间尺寸
        args.PCA = 30  # 论文Fig. 15显示IP使用30个主成分时准确率最高
    elif args.dataset == 'salinas':
        X = sio.loadmat('./oridata/salinas/salinas.mat')['salinas']
        y = sio.loadmat('./oridata/salinas/salinas_gt.mat')['salinas_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 15  # 论文Fig. 14显示15×15是最佳空间尺寸
        args.PCA = 15  # 论文Fig. 15显示SA使用15个主成分时准确率最高
    else:
        raise ValueError(f"不支持的dataset: {args.dataset}")

    print('=' * 60)
    print(f"数据集: {args.dataset}")
    print(f"Hyperspectral data shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"原始波段数: {args.hsi_bands}")
    print(f"PCA降维到: {args.PCA}个主成分")
    print(f"Patch大小: {args.patch_size}×{args.patch_size}")
    print('=' * 60)

    if args.is_train:
        print('\n... ... PCA transformation ... ...')
        X_pca = applyPCA(X, numComponents=args.PCA)
        print(f'Data shape after PCA: {X_pca.shape}')

        print('\n... ... create data cubes ... ...')
        X_cubes, y_cubes = createImageCubes(X_pca, y, windowSize=args.patch_size)
        print(f'Data cube X shape: {X_cubes.shape}')
        print(f'Data cube y shape: {y_cubes.shape}')

        # 添加类别分布检查
        unique, counts = np.unique(y_cubes, return_counts=True)
        print("\n=== 类别分布（模型使用的标签，已减1）===")
        print("原始类别 -> 模型类别")
        for cls, count in zip(unique, counts):
            print(f"原始类别 {int(cls + 1):2d} -> 模型类别 {int(cls):2d}: {count} samples")
        print("=========================================\n")

        print('\n... ... create train & test data ... ...')
        # 分割训练集和测试集
        Xtrain, Xtest, ytrain, ytest = split_data(X_cubes, y_cubes, 1 - args.train_ratio, splitdset="sklearn")
        print(f'Xtrain shape: {Xtrain.shape}')
        print(f'Xtest  shape: {Xtest.shape}')

        # 重塑为 (samples, patch_size, patch_size, PCA, 1)
        Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
        Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.PCA, 1)
        print(f'before transpose: Xtrain shape: {Xtrain.shape}')
        print(f'before transpose: Xtest  shape: {Xtest.shape}')

        # 转换为PyTorch需要的格式 (samples, 1, PCA, patch_size, patch_size)
        Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)  # (N, H, W, C, 1) -> (N, 1, C, H, W)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2)
        print(f'after transpose: Xtrain shape: {Xtrain.shape}')
        print(f'after transpose: Xtest  shape: {Xtest.shape}')

        # 创建DataLoader
        trainset = TrainDS(Xtrain, ytrain)
        testset = TestDS(Xtest, ytest)

        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, test_loader

    else:
        # 测试模式
        print('\n... ... PCA transformation ... ...')
        X_pca = applyPCA(X, numComponents=args.PCA)
        print(f'Data shape after PCA: {X_pca.shape}')

        # 为整个图像padding，用于预测
        X_padded = padWithZeros(X_pca, args.patch_size // 2)
        return X_padded, y