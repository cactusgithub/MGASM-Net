import gc

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
import SimpleITK as sitk
from glob import glob
from scipy.ndimage import distance_transform_edt
from typing import  Tuple, cast
from torch import Tensor
from skimage.morphology import skeletonize
import csv
from scipy.spatial import distance

class AirwayData(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, file, phase, randomCropFlag=False, cropFfactor=(128,128,128),stride=(127,127,127), bodFlag = True,
                 sampleFlag = False, edgeFlag = False,samPointNum = 2,samPointDis=100,datasetflag='',augmentationflag = False,
                 randomCropSamplingFlag = True):

        self.phase = phase
        self.randomCropFlag = randomCropFlag
        self.cropFactor = cropFfactor#代码按zyx写的
        self.stride = stride#24-5-15#代码按zyx写的
        self.bodFlag = bodFlag
        self.edgeFlag = edgeFlag
        self.datasetflag = datasetflag
        self.augmentationflag = augmentationflag
        self.randomCropSamplingFlag = randomCropSamplingFlag



        if phase == "train":
            # ATM22
            if self.datasetflag == "ATM22":
                self.datapath = os.path.join(file,"train(unresampling)","image") # 数据集存放的路径
                self.labelpath = os.path.join(file,"train(unresampling)","label")
                self.edgepath = os.path.join(file,"train(unresampling)","edge")
                self.samPointNum = samPointNum
                self.samPointDis = samPointDis


                print("-------------------------Load all data into memory---------------------------")
                """
                count the number of cases
                """

                self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))
                self.labelFileList = glob(os.path.join(self.labelpath, '*_label*'))

                self.caseNumber = len(self.filelist)

                if len(self.filelist) > 0 and len(self.filelist) == len(self.labelFileList):

                    print("total %s case number: %d" % (self.phase, self.caseNumber))

                    if not self.randomCropFlag:#全部裁剪
                        self.idlist = self.splitDatabypath(self.labelFileList, sampleFlag = sampleFlag)
                        samplepath = os.path.join(self.labelpath, "sampleinfor.csv")
                        with open(samplepath, "a") as csvout:
                            writer = csv.writer(csvout)
                            # writer.writerow(["epoch: ", epoch])
                            # row = ["index", 'totalMetrics', 'TD', 'BD', 'DCS', "accuracy", 'sensitive', 'specificity']
                            for i in range(len(self.idlist)):
                                writer.writerow(self.idlist[i])
                            csvout.close()
                        print("total %s sub train case number: %d" % (self.phase, len(self.idlist) / 2))
                else:
                    print("train data acquisition failure!")

            #BAS
            elif self.datasetflag == 'BAS':
                self.datapath = os.path.join(file, "BAS", "train", "image")  # 数据集存放的路径
                self.labelpath = os.path.join(file,"BAS", "train", "label")
                self.edgepath = os.path.join(file, "BAS", "train", "edge")
                self.samPointNum = samPointNum
                self.samPointDis = samPointDis

                print("-------------------------Load all data into memory---------------------------")
                """
                count the number of cases
                """

                self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))
                self.labelFileList = glob(os.path.join(self.labelpath, '*_label*'))

                self.caseNumber = len(self.filelist)

                if len(self.filelist) > 0 and len(self.filelist) == len(self.labelFileList):

                    print("total %s case number: %d" % (self.phase, self.caseNumber))

                    if not self.randomCropFlag:  # 全部裁剪
                        self.idlist = self.splitDatabypath(self.labelFileList, sampleFlag=sampleFlag)
                        samplepath = os.path.join(self.labelpath, "sampleinfor.csv")
                        with open(samplepath, "a") as csvout:
                            writer = csv.writer(csvout)
                            # writer.writerow(["epoch: ", epoch])
                            # row = ["index", 'totalMetrics', 'TD', 'BD', 'DCS', "accuracy", 'sensitive', 'specificity']
                            for i in range(len(self.idlist)):
                                writer.writerow(self.idlist[i])
                            csvout.close()
                        print("total %s sub train case number: %d" % (self.phase, len(self.idlist) / 2))
                else:
                    print("train data acquisition failure!")


        elif phase =="val":
            # if self.dataflag == "Aeropath":
            #     self.datapath = os.path.join(file, "Aeropath", "image")  # 数据集存放的路径
            #     self.labelpath = os.path.join(file, "Aeropath", "label")
            #     self.edgepath = os.path.join(file, "Aeropath", "edge")
            #
            #     self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))
            #     self.labelFileList = glob(os.path.join(self.labelpath, '*_label*'))  # 24-5-15
            #
            #     self.caseNumber = len(self.filelist)
            #
            #     if len(self.filelist) > 0 and len(self.filelist) == len(self.labelFileList):  # 24-5-15
            #
            #         print("total %s case number: %d" % (self.phase, self.caseNumber))
            #         self.idlist = self.splitDatabypath(self.filelist)
            #         print("total %s sub validation case number: %d" % (self.phase, len(self.idlist) / 2))
            #     else:
            #         print("val data acquisition failure!")
            # else:

            # yuanben
            if self.datasetflag == "ATM22":
                self.datapath = os.path.join(file, "val(unresampling)", "image")  # 数据集存放的路径
                self.labelpath = os.path.join(file, "val(unresampling)", "label")
                self.edgepath = os.path.join(file, "val(unresampling)", "edge")

                self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))
                self.labelFileList = glob(os.path.join(self.labelpath, '*_label*'))  # 24-5-15

                self.caseNumber = len(self.filelist)

                if len(self.filelist) > 0 and len(self.filelist) == len(self.labelFileList):  # 24-5-15

                    print("total %s case number: %d" % (self.phase, self.caseNumber))
                    self.idlist = self.splitDatabypath(self.filelist)
                    print("total %s sub validation case number: %d" % (self.phase, len(self.idlist) / 2))
                else:
                    print("val data acquisition failure!")

            # Aeropath
            elif self.datasetflag == 'Aeropath':
                self.datapath = os.path.join(file, "Aeropath", "image")  # 数据集存放的路径
                self.labelpath = os.path.join(file, "Aeropath", "label")
                self.edgepath = os.path.join(file, "Aeropath", "edge")

                self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))
                self.labelFileList = glob(os.path.join(self.labelpath, '*_label*'))  # 24-5-15

                self.caseNumber = len(self.filelist)

                if len(self.filelist) > 0 and len(self.filelist) == len(self.labelFileList):  # 24-5-15

                    print("total %s case number: %d" % (self.phase, self.caseNumber))
                    self.idlist = self.splitDatabypath(self.filelist)
                    print("total %s sub validation case number: %d" % (self.phase, len(self.idlist) / 2))
                else:
                    print("val data acquisition failure!")
            # BAS
            elif self.datasetflag == 'BAS':
                self.datapath = os.path.join(file,"BAS", "val", "image")  # 数据集存放的路径
                self.labelpath = os.path.join(file,"BAS", "val", "label")
                self.edgepath = os.path.join(file,"BAS", "val", "edge")

                self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))
                self.labelFileList = glob(os.path.join(self.labelpath, '*_label*'))  # 24-5-15

                self.caseNumber = len(self.filelist)

                if len(self.filelist) > 0 and len(self.filelist) == len(self.labelFileList):  # 24-5-15

                    print("total %s case number: %d" % (self.phase, self.caseNumber))
                    self.idlist = self.splitDatabypath(self.filelist)
                    print("total %s sub validation case number: %d" % (self.phase, len(self.idlist) / 2))
                else:
                    print("val data acquisition failure!")
        elif phase == "test":
            if self.datasetflag == 'Aeropath':
                self.datapath = os.path.join(file, "Aeropath", "image")  # 数据集存放的路径
                self.labelpath = os.path.join(file, "Aeropath", "label")
                self.edgepath = os.path.join(file, "Aeropath", "edge")

                self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))
                self.labelFileList = glob(os.path.join(self.labelpath, '*_label*'))  # 24-5-15

                self.caseNumber = len(self.filelist)

                if len(self.filelist) > 0 and len(self.filelist) == len(self.labelFileList):  # 24-5-15

                    print("total %s case number: %d" % (self.phase, self.caseNumber))
                    self.idlist = self.splitDatabypath(self.filelist)
                    print("total %s sub validation case number: %d" % (self.phase, len(self.idlist) / 2))
                else:
                    print("val data acquisition failure!")

            elif self.datasetflag == 'COPD':
                self.datapath = os.path.join(file, "COPD", "image")  # 数据集存放的路径

                self.filelist = glob(os.path.join(self.datapath, '*_clean_hu*'))


                self.caseNumber = len(self.filelist)

                if len(self.filelist) > 0 :  # 24-5-15

                    print("total %s case number: %d" % (self.phase, self.caseNumber))
                    self.idlist = self.splitDatabypath(self.filelist)
                    print("total %s sub validation case number: %d" % (self.phase, len(self.idlist) / 2))
                else:
                    print("val data acquisition failure!")

        else:
            print('mode wrong !')
            ##24-5-15 e




    def __len__(self):
        """
        :return: length of the dataset
        """
        if self.phase == 'train':
            if self.randomCropFlag:
                return self.caseNumber
                # return 1
            else:
                return int(len(self.idlist) / 2)
        elif self.phase == "val" or self.phase == "test":
            return int(len(self.idlist) / 2)
            # return 1

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: image tensor、label tensor 、 label edge tensor
        """
        if self.phase == 'train':
            if self.randomCropFlag:
                # ____________随机裁剪规定大小区域____________________________#
                # # 加载CT图像
                # imgs = sitk.ReadImage(self.filelist[idx])
                # data_name = self.filelist[idx].split('ATM_')[-1].split('_')[0]
                # # 标签
                # labelpath = self.matchingCT2Label(idx,1)
                # labels = sitk.ReadImage(labelpath)
                # # 标签边缘
                # label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)
                if not self.edgeFlag:
                    # 加载CT图像
                    imgs = sitk.ReadImage(self.filelist[idx])
                    data_name = self.filelist[idx].split('ATM_')[-1].split('_')[0]
                    # 标签
                    labelpath = self.matchingCT2Label(idx, 1)
                    labels = sitk.ReadImage(labelpath)
                    # 标签边缘
                    label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)
                    imgs = sitk.GetArrayFromImage(imgs)
                    if self.randomCropSamplingFlag:
                        imgsarr, labelarr = self.randomCropmulti(imgs, label_np, num=self.samPointNum,
                                                                 mindistance=self.samPointDis)
                        return imgsarr, labelarr
                    else:
                        imgs, label_np= self.randomCrop(imgs, label_np)
                        imgs = torch.from_numpy(imgs).to(torch.float32)
                        label_np = torch.from_numpy(label_np).to(torch.float32)

                        return imgs, label_np
                        # imgsarr=[imgs]
                        # labelarr=[label_np]
                        # return imgsarr, labelarr

                # 加载CT图像
                imgs = sitk.ReadImage(self.filelist[idx])
                data_name = self.filelist[idx].split('ATM_')[-1].split('_')[0]
                # 标签
                labelpath = self.matchingCT2Label(idx, 1)
                labels = sitk.ReadImage(labelpath)

                edgepath = self.matchingCT2Label(idx,2)
                labelEdgeImage = sitk.ReadImage(edgepath)
                if self.augmentationflag:
                    if random.randint(0,3) == 3:
                        x = random.randint(-15,15)
                        y = random.randint(-15, 15)
                        z = random.randint(-15, 15)
                        imgs,labels,labelEdgeImage = self.randomRotate(imgs,labels,[x,y,z],labelEdgeImage)
                    elif random.randint(0,3) == 1:
                        f = round(random.uniform(1.0,1.3)-0.1,2)
                        imgs, labels, labelEdgeImage = self.randomScale(imgs, labels, f, labelEdgeImage)

                # 标签边缘
                label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)
                labelEdge = sitk.GetArrayFromImage(labelEdgeImage).astype(np.uint8)
                imgs = sitk.GetArrayFromImage(imgs)
                if self.randomCropSamplingFlag:
                    imgsarr, labelarr,edgearr = self.randomCropmulti(imgs, label_np, labelEdge,self.samPointNum, self.samPointDis)
                else:
                    imgs, label_np, labelEdge = self.randomCropEdge(imgs, label_np, labelEdge)
                    imgsarr = [torch.from_numpy(imgs).to(torch.float32)]
                    labelarr = [torch.from_numpy(label_np).to(torch.float32)]
                    edgearr = [torch.from_numpy(labelEdge).to(torch.float32)]

                if self.bodFlag:
                    # 边界损失 distance map
                    labelEdgeDisMap = self.dist_map_transform([1.0,1.0,1.0], 2, label_np)
                    imgs = torch.from_numpy(imgs).to(torch.float32)
                    label_np = torch.from_numpy(label_np).to(torch.float32)
                    labelEdge = torch.from_numpy(labelEdge).to(torch.float32)  # 返回的张量  #24-5-15
                    return imgs, label_np, labelEdge  ,torch.from_numpy(labelEdgeDisMap).to(torch.float32)
                else:
                    # imgs = torch.from_numpy(imgs).to(torch.float32)
                    # label_np = torch.from_numpy(label_np).to(torch.float32)
                    # labelEdge = torch.from_numpy(labelEdge).to(torch.float32)  # 返回的张量  #24-5-15
                    # return imgs, label_np, labelEdge
                    return imgsarr, labelarr, edgearr

            else:
                #____________非随机裁剪，样本按裁剪大小逐步长裁剪，全部采样_____________________________#
                # 注：没有更改数据采样规则randomCropmulti
                data_name = self.idlist[idx * 2].split('ATM_')[-1].split('_')[0]
                # imagepath = self.matchingCT2Labelbypath(self.idlist[idx * 2],2)
                imagepath = self.dataMatchingbyname(data_name,3)
                imgs = sitk.ReadImage(imagepath)


                # 24-5-15 s
                ip = self.idlist[idx * 2 + 1]
                imgs = sitk.GetArrayFromImage(imgs)[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]

                # 标签

                labels = sitk.ReadImage(self.idlist[idx * 2])
                label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)

                if not self.edgeFlag:
                    label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32)
                # 标签边缘
                edgepath = self.dataMatchingbyname(data_name, 2)
                labelEdgeImage = sitk.ReadImage(edgepath)
                labelEdge = sitk.GetArrayFromImage(labelEdgeImage).astype(np.uint8)
                # labelEdge = self.mask_to_edges(label_np)

                label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]

                if self.bodFlag:
                    # 边界损失 distance map
                    labelEdgeDisMap = self.dist_map_transform([1.0, 1.0, 1.0], 2, label_np)
                    # labelEdgeDisMap = labelEdgeDisMap[:, ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(torch.float32), torch.from_numpy(labelEdgeDisMap).to(
                        torch.float32)
                else:
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(torch.float32)


        elif self.phase == 'val':
            if self.datasetflag == "ATM22" or self.datasetflag == "BAS":
                imgs = sitk.ReadImage(self.idlist[idx * 2])

                data_name = self.idlist[idx * 2].split('ATM_')[-1].split('_')[0]
                # 24-5-15 s
                ip = self.idlist[idx * 2 + 1]
                imgs = sitk.GetArrayFromImage(imgs)[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]

                # 标签
                # labelpath = self.matchingCT2Labelbypath(self.idlist[idx * 2])
                labelpath = self.dataMatchingbyname(data_name, 1)
                labels = sitk.ReadImage(labelpath)
                info = []
                # a =labels.GetSize()
                info.append(labels.GetSize())
                info.append(labels.GetOrigin())
                info.append(labels.GetSpacing())
                info.append(labels.GetDirection())
                label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)

                if not self.edgeFlag:
                    label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), data_name, info, ip
                # 标签边缘
                # labelEdge = self.mask_to_edges(label_np) #numpy格式
                edgepath = self.dataMatchingbyname(data_name, 2)
                labelEdgeImage = sitk.ReadImage(edgepath)
                labelEdge = sitk.GetArrayFromImage(labelEdgeImage).astype(np.uint8)

                label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]

                if self.bodFlag:
                    # 边界损失 distance map
                    labelEdgeDisMap = self.dist_map_transform([1.0, 1.0, 1.0], 2, label_np)
                    # labelEdgeDisMap = labelEdgeDisMap[:, ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(
                        torch.float32), torch.from_numpy(labelEdgeDisMap).to(
                        torch.float32), data_name, info, ip
                else:
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(
                        torch.float32), data_name, info, ip

            #Aeropath
            elif self.datasetflag == 'Aeropath':
                imgs = sitk.ReadImage(self.idlist[idx * 2])
                data_name = self.idlist[idx * 2].split('/')[-1].split('_')[0]
                # 24-5-15 s
                ip = self.idlist[idx * 2 + 1]
                imgs = sitk.GetArrayFromImage(imgs)[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                # 标签
                # labelpath = self.matchingCT2Labelbypath(self.idlist[idx * 2])
                labelpath = self.dataMatchingbyname4Are(data_name, 1)
                labels = sitk.ReadImage(labelpath)
                info = []
                # a =labels.GetSize()
                info.append(labels.GetSize())
                info.append(labels.GetOrigin())
                info.append(labels.GetSpacing())
                info.append(labels.GetDirection())
                label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)
                if not self.edgeFlag:
                    label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), data_name, info, ip
                # 标签边缘
                # labelEdge = self.mask_to_edges(label_np) #numpy格式
                edgepath = self.dataMatchingbyname4Are(data_name, 2)
                labelEdgeImage = sitk.ReadImage(edgepath)
                labelEdge = sitk.GetArrayFromImage(labelEdgeImage).astype(np.uint8)
                label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                if self.bodFlag:
                    # 边界损失 distance map
                    labelEdgeDisMap = self.dist_map_transform([1.0, 1.0, 1.0], 2, label_np)
                    # labelEdgeDisMap = labelEdgeDisMap[:, ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(
                        torch.float32), torch.from_numpy(labelEdgeDisMap).to(
                        torch.float32), data_name, info, ip
                else:
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(
                        torch.float32), data_name, info, ip

        elif self.phase=="test":
            if self.datasetflag == 'Aeropath':
                imgs = sitk.ReadImage(self.idlist[idx * 2])
                data_name = self.idlist[idx * 2].split('/')[-1].split('_')[0]
                # 24-5-15 s
                ip = self.idlist[idx * 2 + 1]
                imgs = sitk.GetArrayFromImage(imgs)[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                # 标签
                # labelpath = self.matchingCT2Labelbypath(self.idlist[idx * 2])
                labelpath = self.dataMatchingbyname4Are(data_name,1)
                labels = sitk.ReadImage(labelpath)
                info = []
                # a =labels.GetSize()
                info.append(labels.GetSize())
                info.append(labels.GetOrigin())
                info.append(labels.GetSpacing())
                info.append(labels.GetDirection())
                label_np = sitk.GetArrayFromImage(labels).astype(np.uint8)
                if not self.edgeFlag:
                    label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), data_name, info, ip
                # 标签边缘
                # labelEdge = self.mask_to_edges(label_np) #numpy格式
                edgepath = self.dataMatchingbyname4Are(data_name, 2)
                labelEdgeImage = sitk.ReadImage(edgepath)
                labelEdge = sitk.GetArrayFromImage(labelEdgeImage).astype(np.uint8)
                label_np = label_np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                if self.bodFlag:
                    # 边界损失 distance map
                    labelEdgeDisMap = self.dist_map_transform([1.0, 1.0, 1.0], 2, label_np)
                    # labelEdgeDisMap = labelEdgeDisMap[:, ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(
                        torch.float32), torch.from_numpy(labelEdgeDisMap).to(
                        torch.float32), data_name, info, ip
                else:
                    labelEdge = labelEdge[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]
                    return torch.from_numpy(imgs).to(torch.float32), torch.from_numpy(label_np).to(
                        torch.float32), torch.from_numpy(labelEdge).to(
                        torch.float32), data_name, info, ip

            elif self.datasetflag == 'COPD':
                image = sitk.ReadImage(self.idlist[idx * 2])
                data_name = self.idlist[idx * 2].split('/')[-1].split('_')[0]
                # 24-5-15 s
                ip = self.idlist[idx * 2 + 1]
                imgs = sitk.GetArrayFromImage(image)[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]]


                info = []
                # a =labels.GetSize()
                info.append(image.GetSize())
                info.append(image.GetOrigin())
                info.append(image.GetSpacing())
                info.append(image.GetDirection())



                return torch.from_numpy(imgs).to(torch.float32),  data_name, info, ip

    def splitDatabypath(self,imagepathlist, sampleFlag = False):
        if imagepathlist is not None:
            # idlist=[]
            imgsList = []
            for index in range(len(imagepathlist)):
                image = sitk.GetArrayFromImage(sitk.ReadImage(imagepathlist[index]))
                z, y, x = image.shape

                # for index in range(image)
                for iz in range(0, z, self.stride[0]):
                    if iz + self.cropFactor[0] > z - 1:
                        iz = z - self.cropFactor[0]

                    for iy in range(0, y, self.stride[1]):
                        if iy + self.cropFactor[1] > y - 1:
                            iy = y - self.cropFactor[1]

                        for ix in range(0, x, self.stride[2]):
                            if ix + self.cropFactor[2] > x - 1:
                                ix = x - self.cropFactor[2]


                            id = [iz,iz + self.cropFactor[0], iy, iy + self.cropFactor[1], ix, ix + self.cropFactor[2]]

                            if sampleFlag:
                                if self.dataSampling(imagepathlist[index],id,0.15,0.1):
                                    imgsList.append(imagepathlist[index])
                                    imgsList.append(id)
                            else:
                                imgsList.append(imagepathlist[index])
                                imgsList.append(id)


            return imgsList#, labelList, labelEdgeList, numidList

    def dataSampling(self,labelpath,id,ratioSke = 0.15, ratioV = 0.1):
        image = sitk.ReadImage(labelpath)
        imagenp = sitk.GetArrayFromImage(image)

        imagenpSampling = imagenp[id[0]:id[1],id[2]:id[3],id[4]:id[5]]
        skeImage = skeletonize(imagenp)
        skeImageS = skeletonize(imagenpSampling)

        skeNum = np.sum(skeImage, dtype=np.float64)
        skeNumS = np.sum(skeImageS, dtype=np.float64)

        num = np.sum(imagenp, dtype=np.float64)
        numS = np.sum(imagenpSampling, dtype=np.float64)

        if skeNum>0.0:
            if skeNumS / skeNum >= ratioSke:
                return True
        if num > 0.0:
            if numS / num >= ratioV:
                return True

        return False




    # 24-5-15 e

    def matchingCT2Label(self, i,tag=1):
        """
        i为CT image的索引
        tag=1,根据id索引寻找对应的label
        tag=2,根据id索引寻找对应的edge

         """

        CTnum = self.filelist[i].split('ATM_')[-1].split('_')[0]
        if tag ==1:
            labelname = "ATM_"+CTnum+"_0000_label.nii.gz"
            path = glob(os.path.join(self.labelpath,labelname))
            return path[0]
        elif tag == 2:
            edgename = "ATM_"+CTnum+"_0000_edge.nii.gz"
            path = glob(os.path.join(self.edgepath, edgename))
            return path[0]



    def dataMatchingbyname(self,dataname,tag=1):
        """

        :param dataname:
        :param tag: 1：根据dataname寻找对应的label；2：根据dataname寻找对应的edge;3：根据dataname寻找对应的image
        :return:
        """
        if tag ==1:
            labelname = "ATM_"+dataname+"_0000_label.nii.gz"
            path = glob(os.path.join(self.labelpath,labelname))
            return path[0]
        elif tag == 2:
            edgename = "ATM_"+dataname+"_0000_edge.nii.gz"
            path = glob(os.path.join(self.edgepath, edgename))
            return path[0]
        elif tag == 3:
            imagename = "ATM_"+dataname+"_0000_clean_hu.nii.gz"
            path = glob(os.path.join(self.datapath, imagename))
            return path[0]

    def dataMatchingbyname4Are(self,dataname,tag=1):
        """

        :param dataname:
        :param tag: 1：根据dataname寻找对应的label；2：根据dataname寻找对应的edge;3：根据dataname寻找对应的image
        :return:
        """
        if tag ==1:
            labelname = dataname+"_CT_HR_labels.nii.gz"
            path = glob(os.path.join(self.labelpath,labelname))
            return path[0]
        elif tag == 2:
            edgename = dataname+"_CT_HR_edge.nii.gz"
            path = glob(os.path.join(self.edgepath, edgename))
            return path[0]
        elif tag == 3:
            imagename = dataname+"_CT_HR_clean_hu.nii.gz"
            path = glob(os.path.join(self.datapath, imagename))
            return path[0]

    def matchingCT2Labelbypath(self, CTpath,tag=1):
        """
        在label列表中找到与CT对应的标签
        tag=1:根据ct路径选择label路径
        tag!=1:根据标签路径选择ct路径

         """

        CTnum = CTpath.split('ATM_')[-1].split('_')[0]
        if tag ==1:
            for j in range(self.caseNumber):
                labelNum = self.labelFileList[j].split('ATM_')[-1].split('_')[0]
                if CTnum == labelNum:
                    return self.labelFileList[j]
        else:
            for j in range(self.caseNumber):
                labelNum = self.filelist[j].split('ATM_')[-1].split('_')[0]
                if CTnum == labelNum:
                    return self.filelist[j]

    # 提取edge
    # def mask_to_edges(self,mask):
    #     _edge = mask
    #     _edge = self.mask_to_onehot(_edge, 1)
    #     _edge = self.onehot_to_binary_edges(_edge, radius=1, num_classes=1)
    #     return _edge
    #
    #
    # def mask_to_onehot(self,mask, num_classes=1):
    #     _mask = [mask == i for i in range(1, num_classes + 1)]  # 每一类上判断true或false
    #     _mask = [np.expand_dims(x, 0) for x in _mask]  # 将每一个类别扩展一个维度
    #     return np.concatenate(_mask, 0)  # 在0维度上连接每一个类别，形成onehot (num_classes, height, width)
    #
    # def onehot_to_binary_edges(self,mask, radius=2, num_classes=1):
    #     if radius < 0:
    #         return mask
    #
    #     # We need to pad the borders for boundary conditions
    #     mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    #
    #     edgemap = np.zeros(mask.shape[1:])
    #
    #     for i in range(num_classes):
    #         dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
    #         dist = dist[1:-1, 1:-1, 1:-1]
    #         dist[dist > radius] = 0
    #         edgemap += dist
    #     # edgemap = np.expand_dims(edgemap, axis=0)
    #     edgemap = (edgemap > 0).astype(np.uint8)
    #
    #     return edgemap
    #
    # # for boundary loss, 计算distance map
    # def  dist_map_transform(self,resolution, K, label) :
    #     t = self.gt_transform(resolution,K,label)
    #     t = t.cpu().numpy()
    #     t = self.one_hot2dist(t,resolution)
    #
    #     return t
    #
    # def gt_transform(self,resolution, K, label) :
    #
    #     if not isinstance(label,np.ndarray):
    #         label = np.array(label)
    #     label = label[...]
    #     # label = torch.tensor(label,dtype=torch.int32)[None,...]
    #     label = torch.tensor(label, dtype=torch.int32)
    #     label = self.class2one_hot(label,K)
    #     # return itemgetter(0)(label)
    #     return label
    #
    # def one_hot2dist(self, seg, resolution = None,
    #                  dtype=None) :
    #     assert self.one_hot(torch.tensor(seg), axis=0)
    #     K: int = len(seg)
    #
    #     res = np.zeros_like(seg, dtype=dtype)
    #     for k in range(K):
    #         posmask = seg[k].astype(np.bool_)
    #
    #         if posmask.any():
    #             negmask = ~posmask
    #             res[k] = distance_transform_edt(negmask, sampling=resolution) * negmask \
    #                      - (distance_transform_edt(posmask, sampling=resolution) - 1) * posmask
    #
    #         # The idea is to leave blank the negative classes
    #         # since this is one-hot encoded, another class will supervise that pixel
    #     # return res
    #     return np.abs(res)
    #
    #
    # def simplex(self, t, axis=0):
    #     """
    #     是用于检查一个张量在指定轴上是否满足单纯形条件的函数。在这里，单纯形条件指的是张量在该轴上的元素之和为1。
    #     即判断是不是one-hot编码
    #     :param t:
    #     :param axis:
    #     :return:
    #     """
    #     _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    #     _ones = torch.ones_like(_sum, dtype=torch.float32)
    #     flag = torch.allclose(_sum, _ones)
    #     return flag
    #     # return torch.allclose(_sum, _ones)
    #
    # def class2one_hot(self, seg, K):
    #     # Breaking change but otherwise can't deal with both 2d and 3d
    #     # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]
    #
    #     assert self.sset(seg, list(range(K))), (self.uniq(seg), K)
    #
    #     img_shape = tuple(seg.shape)  # type: Tuple[int, ...]
    #
    #     device = seg.device
    #     seg = seg.to(torch.int64)
    #     # 创建onehot编码，背景10 前景01
    #     res = torch.zeros((K, *img_shape), dtype=torch.int32, device=device).scatter_(0, seg[ None, ...], 1)
    #
    #     assert res.shape == (K, *img_shape)
    #     assert self.one_hot(res)
    #
    #     return res
    #
    # def one_hot(self, t, axis=0) :
    #     """
    #     通过检查单纯性和元素是否为01判断是否为onehot编码
    #     :param t:
    #     :param axis:
    #     :return:
    #     """
    #     return self.simplex(t, axis) and self.sset(t, [0, 1])
    #
    # def sset(self, a, sub):
    #     """
    #     判断a中的元素是不是sub的子集，及通过检查元素是不是只包含0 1判断是不是onehot编码
    #     :param a:
    #     :param sub:
    #     :return:
    #     """
    #
    #     return self.uniq(a).issubset(sub)
    #
    # def uniq(self, a) :
    #     """
    #     提取a中包含的元素值
    #     :param a:
    #     :return:
    #     """
    #     return set(torch.unique(a.cpu()).numpy())


    # for random crop ，默认128
    def randomCrop(self,image,label):
        """
                Make a random crop of the whole volume
                :param image:
                :param label:
                :param cropFactor: The crop size that you want to crop
                :return:
                """
        w, h, d = image.shape
        # print(w, cropFactor[0], h, cropFactor[1], d, cropFactor[2])
        z = random.randint(0, w - self.cropFactor[0])
        y = random.randint(0, h - self.cropFactor[1])
        x = random.randint(0, d - self.cropFactor[2])

        image = image[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
        label = label[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
        return image, label

    def randomCropmulti(self, image, label, labelEdge=None, num=5, mindistance=30):
        """
        根据采样规则随机裁剪5个骨架点数据，返回一骨架点为中心的数据区域
                Make a random crop of the whole volume
                :param image:
                :param label:
                :param cropFactor: The crop size that you want to crop
                :return:
                """
        w, h, d = image.shape
        skeLabel = skeletonize(label)
        targeIndices = np.argwhere(skeLabel == 1)
        skenum = len(targeIndices)

        if skenum < num:
            raise ValueError(f"骨架点数量小于{num}")
        # targeIndices1 = np.array([[1,2,3],[4,5,6],[7,8,9],[11,11,22]])
        np.random.shuffle(targeIndices)
        spoints = []
        imagearr = []
        labelarr = []

        if labelEdge is not None:
            edgearr=[]
        compeletflag = False
        while not compeletflag:
            pointnum = random.randint(0, skenum - 1)
            point = targeIndices[pointnum]

            if len(spoints) == 0:
                z = int(point[0] - self.cropFactor[0] / 2)
                y = int(point[1] - self.cropFactor[1] / 2)
                x = int(point[2] - self.cropFactor[2] / 2)

                if z < 0:
                    z = 0
                elif z + self.cropFactor[0] > w:
                    z = w - self.cropFactor[0]
                if y < 0:
                    y = 0
                elif y + self.cropFactor[1] > h:
                    y = h - self.cropFactor[1]
                if x < 0:
                    x = 0
                elif x + self.cropFactor[2] > d:
                    x = d - self.cropFactor[2]
                id = [z, z + self.cropFactor[0], y, y + self.cropFactor[1], x, x + self.cropFactor[2]]
                if self.dataSampling4randomcrop(label, id, skeLabel):
                    spoints.append(point)
                    i = image[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
                    l = label[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
                    imagearr.append(torch.from_numpy(i).to(torch.float32))
                    labelarr.append(torch.from_numpy(l).to(torch.float32))

                    if labelEdge is not None:
                        e = labelEdge[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
                        edgearr.append(torch.from_numpy(e).to(torch.float32))

                if len(spoints) == num:
                    compeletflag = True


            else:
                distances = [distance.euclidean(point, p) for p in spoints]
                if all(od > mindistance for od in distances):
                    z = int(point[0] - self.cropFactor[0] / 2)
                    y = int(point[1] - self.cropFactor[1] / 2)
                    x = int(point[2] - self.cropFactor[2] / 2)

                    if z < 0:
                        z = 0
                    elif z + self.cropFactor[0] > w:
                        z = w - self.cropFactor[0]
                    if y < 0:
                        y = 0
                    elif y + self.cropFactor[1] > h:
                        y = h - self.cropFactor[1]
                    if x < 0:
                        x = 0
                    elif x + self.cropFactor[2] > d:
                        x = d - self.cropFactor[2]
                    id = [z, z + self.cropFactor[0], y, y + self.cropFactor[1], x, x + self.cropFactor[2]]
                    if self.dataSampling4randomcrop(label, id, skeLabel):
                        spoints.append(point)
                        i = image[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
                        l = label[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
                        imagearr.append(torch.from_numpy(i).to(torch.float32))
                        labelarr.append(torch.from_numpy(l).to(torch.float32))

                        if labelEdge is not None:
                            e = labelEdge[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
                            edgearr.append(torch.from_numpy(e).to(torch.float32))
                    # spoints.append(point)
                if len(spoints) == num:
                    compeletflag = True

        # if len(spoints) < 0:
        #     raise ValueError("无法找到满足距离条件的骨架点")
        if labelEdge is not None:
            return imagearr,labelarr,edgearr
        return imagearr, labelarr

    def dataSampling4randomcrop(self, imagenp, id, skeImage = None,ratioSke=0.15, ratioV=0.1):
        """
        用于在随机裁剪数据样本时，检测裁剪的数据是否符合标准
        :param imagenp:
        :param id:
        :param ratioSke:
        :param ratioV:
        :return:
        """
        # image = sitk.ReadImage(labelpath)
        # imagenp = sitk.GetArrayFromImage(image)

        imagenpSampling = imagenp[id[0]:id[1], id[2]:id[3], id[4]:id[5]]
        if skeImage is None:
            skeImage = skeletonize(imagenp)
        # skeImageS = skeletonize(imagenpSampling)
        skeImageS = skeImage[id[0]:id[1], id[2]:id[3], id[4]:id[5]]

        skeNum = np.sum(skeImage, dtype=np.float64)
        skeNumS = np.sum(skeImageS, dtype=np.float64)

        num = np.sum(imagenp, dtype=np.float64)
        numS = np.sum(imagenpSampling, dtype=np.float64)

        if skeNum > 0.0:
            if skeNumS / skeNum >= ratioSke:
                return True
        if num > 0.0:
            if numS / num >= ratioV:
                return True

        return False

    def randomCropEdge(self, image, label, labelEdge, bodFalg=False):
        """
                Make a random crop of the whole volume
                :param image:
                :param label:
                :param cropFactor: The crop size that you want to crop
                :return:
                """
        w, h, d = image.shape
        # print(w, cropFactor[0], h, cropFactor[1], d, cropFactor[2])
        z = random.randint(0, w - self.cropFactor[0])
        y = random.randint(0, h - self.cropFactor[1])
        x = random.randint(0, d - self.cropFactor[2])

        image = image[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
        label = label[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
        if bodFalg:
            labelEdge = labelEdge[:, z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
        else:
            labelEdge = labelEdge[z:z + self.cropFactor[0], y:y + self.cropFactor[1], x:x + self.cropFactor[2]]
        return image, label, labelEdge

    def randomRotate(self,image,label,angle=[0,0,0],labeledge =None):
        # 定义旋转中心
        center = np.array(image.GetSize()) / 2.0 + np.array(image.GetOrigin())

        # 定义变换
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center)
        transform.SetRotation(angle[0]* np.pi / 180, angle[1] * np.pi / 180, angle[2] * np.pi / 180)

        # 应用变换
        reference_image = image
        rotated_image = sitk.Resample(image, reference_image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())
        rotated_label = sitk.Resample(label, reference_image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())
        if labeledge is not  None:
            rotated_edge = sitk.Resample(labeledge, reference_image, transform, sitk.sitkNearestNeighbor, 0.0, image.GetPixelID())
            return rotated_image, rotated_label, rotated_edge

        return rotated_image,rotated_label
    def randomScale(self,image,label,scaleFactor=1.0,labeledge = None):
        # 定义变换
        transform = sitk.AffineTransform(3)  # 3D变换
        transform.Scale(scaleFactor)
        center = np.array(image.GetSize()) / 2.0 + np.array(image.GetOrigin())
        transform.SetCenter(center)

        # 定义参考图像

        reference_image = image
        # 应用变换
        scaled_image = sitk.Resample(image, reference_image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())
        scaled_label = sitk.Resample(label, reference_image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())
        if labeledge is not  None:
            scaled_edge = sitk.Resample(labeledge, reference_image, transform, sitk.sitkNearestNeighbor, 0.0, image.GetPixelID())
            return scaled_image, scaled_label,scaled_edge

        return scaled_image, scaled_label

