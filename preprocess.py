import os.path

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
def resampleimage(inputpath,outpath,newspace=[1.0,1.0,1.0],newSize=[256,256,256],tag=1,interpolation='bsp'):
    """
    重采样图像到新的维度和空间间距
    tag=1为输入维度大小，不等于1为输入体素间距大小

    by xuyi
    """
    oimage = sitk.ReadImage(inputpath)
    originalSize = oimage.GetSize()
    originalSpace = oimage.GetSpacing()
    originalDirection = oimage.GetDirection()
    originalOrigin = oimage.GetOrigin()
    if tag==1:
        newspace = [osz*ospc/nspc for osz, ospc, nspc in zip(originalSize, originalSpace, newSize)]
    else:
        #计算新的图像维度
        newSize = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(originalSize, originalSpace, newspace)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(oimage)
    # resampler.SetTransform(sitk.Transform(3,sitk.sitkIdentity))
    resampler.SetOutputSpacing(newspace)
    resampler.SetSize(newSize)
    if interpolation =='bsp':
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif interpolation =='near':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation == 'label':
        resampler.SetInterpolator(sitk.sitkLabelGaussian)
    outImage = resampler.Execute(oimage)
    outImage.SetDirection(originalDirection)
    outImage.SetOrigin(originalOrigin)

    sitk.WriteImage(outImage,outpath)

# 提取edge________________________________________________________________
def mask_to_edges(mask):
    _edge = mask
    _edge = mask_to_onehot(_edge, 1)
    _edge = onehot_to_binary_edges(_edge, radius=1, num_classes=1)
    return _edge


def mask_to_onehot(mask, num_classes=1):
    _mask = [mask == i for i in range(1, num_classes + 1)]  # 每一类上判断true或false
    _mask = [np.expand_dims(x, 0) for x in _mask]  # 将每一个类别扩展一个维度
    return np.concatenate(_mask, 0)  # 在0维度上连接每一个类别，形成onehot (num_classes, height, width)

def onehot_to_binary_edges(mask, radius=2, num_classes=1):
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def extractEdge(labelpath,labelEdgePath):
    labels = sitk.ReadImage(labelpath)
    label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)
    labelEdge = mask_to_edges(label_np)
    labelEdgeImage = sitk.GetImageFromArray(labelEdge)
    labelEdgeImage.CopyInformation(labels)
    sitk.WriteImage(labelEdgeImage,labelEdgePath)
    return labelEdge
# end edge__________________________________________________________________


## for boundary loss, 计算distance map
import torch
from typing import  Tuple, cast
from torch import Tensor
def  dist_map_transform(resolution, K, label) :
    t = gt_transform(K,label)
    t = t.cpu().numpy()
    t = one_hot2dist(t,resolution)

    return t

def gt_transform( K, label) :

    if not isinstance(label,np.ndarray):
        label = np.array(label)
    label = label[...]
    # label = torch.tensor(label,dtype=torch.int32)[None,...]
    label = torch.tensor(label, dtype=torch.int32)
    label = class2one_hot(label,K)
    # return itemgetter(0)(label)
    return label

def one_hot2dist( seg, resolution = None,
                 dtype=None) :
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool_)

        if posmask.any():
            negmask = ~posmask
            res[k] = distance_transform_edt(negmask, sampling=resolution) * negmask \
                     - (distance_transform_edt(posmask, sampling=resolution) - 1) * posmask

        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel
    # return res
    return np.abs(res)


def simplex( t, axis=0):
    """
    是用于检查一个张量在指定轴上是否满足单纯形条件的函数。在这里，单纯形条件指的是张量在该轴上的元素之和为1。
    即判断是不是one-hot编码
    :param t:
    :param axis:
    :return:
    """
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    flag = torch.allclose(_sum, _ones)
    return flag
    # return torch.allclose(_sum, _ones)

def class2one_hot( seg, K):
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    img_shape = tuple(seg.shape)  # type: Tuple[int, ...]

    device = seg.device
    seg = seg.to(torch.int64)
    # 创建onehot编码，背景10 前景01
    res = torch.zeros((K, *img_shape), dtype=torch.int32, device=device).scatter_(0, seg[ None, ...], 1)

    assert res.shape == (K, *img_shape)
    assert one_hot(res)

    return res

def one_hot( t, axis=0) :
    """
    通过检查单纯性和元素是否为01判断是否为onehot编码
    :param t:
    :param axis:
    :return:
    """
    return simplex(t, axis) and sset(t, [0, 1])

def sset( a, sub):
    """
    判断a中的元素是不是sub的子集，及通过检查元素是不是只包含0 1判断是不是onehot编码
    :param a:
    :param sub:
    :return:
    """

    return uniq(a).issubset(sub)

def uniq( a) :
    """
    提取a中包含的元素值
    :param a:
    :return:
    """
    return set(torch.unique(a.cpu()).numpy())

from skimage.morphology import skeletonize
def comdistance2ske4qin(labelpath,skepath,dispath,m=0.1,rd=0.5):
    label = sitk.ReadImage(labelpath)
    labelnp = sitk.GetArrayFromImage(label)
    skenp = skeletonize(labelnp)

    distancemap = distance_transform_edt(1-skenp)
    positionOfback = np.where(labelnp == 0)
    positionOfFore =  np.where(labelnp > 0)
    distancemap[positionOfback] = 0
    maxvalue = np.amax(distancemap)
    if maxvalue > 0:
        for position in zip(*positionOfFore):
            di = distancemap[position]
            if di > 0:
                distancemap[position] = 1 - m * (di / maxvalue) ** rd
    distancemap[positionOfback]  == 1
    distancemapsitk = sitk.GetImageFromArray(distancemap)
    skenp = skenp.astype(dtype='uint8')
    ske = sitk.GetImageFromArray(skenp)
    ske.CopyInformation(label)
    distancemapsitk.CopyInformation(label)
    sitk.WriteImage(ske,skepath)
    sitk.WriteImage(distancemapsitk,dispath)

from glob import glob
if __name__ == '__main__':
    # labelpath = "/home/xy/pyProjects/TubularStructureSeg/Data/val(unresampling)/label"
    # labelpath = "/home/xy/pyProjects/TubularStructureSeg/Data/Aeropath/label"
    # labellist = glob(os.path.join(labelpath,"*label*"))
    # numlabel = len(labellist)
    # for i in range(numlabel):
        # num = labellist[i].split("ATM_")[-1].split("_")[0]
        # skepath = "/home/xy/pyProjects/TubularStructureSeg/Data/val(unresampling)/skedis/ATM_"+num+"_ske.nii.gz"
        # distancemap = "/home/xy/pyProjects/TubularStructureSeg/Data/val(unresampling)/skedis/ATM_"+num+"_distancemap.nii.gz"

        # Aeropath
        # num = labellist[i].split("_CT")[0].split("/")[-1]
        # skepath = "/home/xy/pyProjects/TubularStructureSeg/Data/Aeropath/skedis/" + num + "_CT_HR_ske.nii.gz"
        # distancemap = "/home/xy/pyProjects/TubularStructureSeg/Data/Aeropath/skedis/" + num + "_CT_HR_distancemap.nii.gz"
        # comdistance2ske4qin(labellist[i],skepath,distancemap)

    # inputpath = './data/fixed4/fixed4.nii.gz'
    # labelpath = './data/fixed4/segfix4.nii'
    # outpath = './data/fixed4/lung4.nii.gz'

    # cropCT(inputpath,outpath,labelpath)
    # num =4
    # print(f"{num}")
    #
    # inputpath = "./data/ATM_001_0000_clean_hu.nii.gz"
    # outpath= "./data/r2(size128128128 no transform bspline).nii.gz"
    # newsapce = [1.5, 1.5, 1.5]
    # newsize = []
    # resampleimage(inputpath,outpath,tag=1)
    import  os
    from glob import glob
    # file = './Data/BAS/val/label1'
    file = "/dev/raid/doctor/xy/DataSets/Parse22/label/"
    # file = 'G:\\WorkSpace\\MRF\\LungAirwaySeg\\TubularStructureSeg\\Data\\train(unresampling)\\label'
    # datapath = os.path.join(file, "train(unresampling)", "image")  # 数据集存放的路径
    # labelpath = os.path.join(file, "Aeropath", "label")
    labellist = glob(os.path.join(file,"*label*"))
    # skepath = "./Data/BAS/val/edge"
    num = len(labellist)
    skepath ="/dev/raid/doctor/xy/DataSets/Parse22/edge/"
    for i in range(num):
        # skeName = labellist[i].split('/')[-1].split('_')[0] +"_CT_HR_edge.nii.gz"
        # skeName = labellist[i].split('ATM_')[-1].split('_')[0] + "_0000_edge.nii.gz"
        skeName = labellist[i].split('_label')[0].split('/')[-1]+'_edge.nii.gz'
        skeoutpath = os.path.join(skepath,skeName)
        extractEdge(labellist[i],skeoutpath)





    # print("-------------------------Load all data into memory---------------------------")
    # """
    # count the number of cases
    # """
    #
    # # allimgdata_memory = {}
    # # alllabeldata_memory = {}
    #
    # filelist = glob(os.path.join(datapath, '*_clean_hu*'))
    # labelFileList = glob(os.path.join(labelpath, '*_label*'))
    #
    # outimagepath = os.path.join(file,"train(space111)","image")
    # outlabelpath = os.path.join(file,'train(space111)',"label_label")
    #
    # # if len(labelFileList)==280:
    # if len(filelist) == len(labelFileList):
    # #     num = len(filelist)
    #
    #     num = len(labelFileList)
    #     for i in range(num):
    #         imageName = "ATM_"+filelist[i].split('ATM_')[-1]
    #         labelName = "ATM_"+labelFileList[i].split('ATM_')[-1]
    #         outimagepath1 = os.path.join(outimagepath,imageName)
    #         outlabelpath1 = os.path.join(outlabelpath,labelName)
    #         # resampleimage(filelist[i],outimagepath1,tag=0,interpolation='bsp')
    #         resampleimage(labelFileList[i],outlabelpath1,tag=0,interpolation='label')
    #         print(f"完成{i}")
    #
    # else:
    #     # print(f"image's length is {len(filelist)},label's length is {len(labelFileList)}")
    #     print(f"label's length is {len(labelFileList)}")