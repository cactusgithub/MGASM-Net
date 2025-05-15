import csv
import os.path

from torch.nn.init import xavier_normal_, kaiming_normal_, constant_, normal_
import torch.nn as nn
from models import norm as mynn

import numpy as np
import SimpleITK as sitk
import evaluation_atm_22 as atm22
from skimage.morphology import skeletonize,cube,closing
from skimage import measure



def weights_init(net, init_type='normal'):
    """
    :param m: modules of CNNs
    :return: initialized modules
    """
    def init_func(m):
        # if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        if isinstance(m, nn.Conv3d) :
            if init_type == 'normal':
                normal_(m.weight.data)
            elif init_type == 'xavier':
                xavier_normal_(m.weight.data)
            else:
                kaiming_normal_(m.weight.data)
            if m.bias is not None:
                constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            # 对于归一化层，通常初始化权重为1，偏置为0
            constant_(m.weight, 1)
            constant_(m.bias, 0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return

def combineImage(predList,valpath,inputLabelPath,skepath,epoch=0,datasetflag="",featureNum=3):
    """

    :param predList:
    列表元素：   tempDics = {"name": name, "info": labelInfo, "sub": subVolumList}
    info:
    info=[]
            info.append(labels.GetSize())
            info.append(labels.GetOrigin())
            info.append(labels.GetSpacing())
            info.append(labels.GetDirection())
    subVolumList = [siglabel, subPosition, prelabeledges]
    subPosition为裁剪区域在原图中的位置，原图为numpy数组类型zyx subposition 列表内为张量
    :return:
    """
    if isinstance(predList, list):
        if not os.path.exists(valpath):
            os.makedirs(valpath)
        errpath = os.path.join(valpath,"valmetrics.csv")
        with open(errpath, "a") as csvout:
            writer = csv.writer(csvout)
            writer.writerow(["epoch: ", epoch])
            row = ["index", 'totalMetrics', 'TD', 'BD', 'DCS', "accuracy", 'sensitive', 'specificity']
            writer.writerow(row)
            csvout.close()

        metricsList = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        numoflist = len(predList)

        for i in range(len(predList)):

            infolist = predList[i]['info']
            labelnp = np.zeros(infolist[0][::-1])
            # edgenp = np.zeros(infolist[0][::-1])

            tagnp = np.zeros(infolist[0][::-1])
            subVolum = predList[i]['sub']

            for j in range(int(len(subVolum)/featureNum)):
                ip = subVolum[j*featureNum+1]
                labelnp[ip[0]:ip[1],ip[2]:ip[3],ip[4]:ip[5]] += subVolum[j*featureNum]
                # edgenp[ip[0]:ip[1],ip[2]:ip[3],ip[4]:ip[5]] += subVolum[j*featureNum+2]

                tagnp[ip[0]:ip[1],ip[2]:ip[3],ip[4]:ip[5]] +=1
            if np.all(tagnp != 0):
                labelnp1 = labelnp / tagnp
                # edgenp1 = edgenp / tagnp
                # labeladdedgenp1 = labelnp1 + edgenp1
                # return labelnp
                labelnp = (labelnp1 > 0.5).astype(dtype='uint8')
                # edgenp = (edgenp1 > 0.5).astype(dtype='uint8')
                # labeladdedgenp = (labeladdedgenp1 > 0.5).astype(dtype='uint8')
                # labelnp = (labelnp > 0.8).astype(dtype='uint8')

                # 提取最大连通分量
                labelnp = maxConnect(labelnp)

                labelimage = sitk.GetImageFromArray(labelnp)
                labelimage1 = sitk.GetImageFromArray(labelnp1)
                # edgeimage = sitk.GetImageFromArray(edgenp)
                # edgeimage1 = sitk.GetImageFromArray(edgenp1)
                # labeladdedgeImage = sitk.GetImageFromArray(labeladdedgenp)
                # labeladdedgeImage1 = sitk.GetImageFromArray(labeladdedgenp1)
                # a = [t.item() for t in infolist[1]]
                labelimage.SetOrigin([t.item() for t in infolist[1]])
                labelimage.SetSpacing([t.item() for t in infolist[2]])
                labelimage.SetDirection([t.item() for t in infolist[3]])
                labelimage1.CopyInformation(labelimage)
                # edgeimage.CopyInformation(labelimage)
                # edgeimage1.CopyInformation(labelimage)
                # labeladdedgeImage.CopyInformation(labelimage)
                # labeladdedgeImage1.CopyInformation(labelimage)

                # if not os.path.exists(valpath):
                #     os.makedirs(valpath)
                path = os.path.join(valpath,f"ATM_{predList[i]['name']}_pre.nii.gz")#'./result/val/'+predList[i][name] +"nii.gz"
                path1 = os.path.join(valpath,
                                    f"ATM_{predList[i]['name']}_pre1.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                sitk.WriteImage(labelimage,path)
                sitk.WriteImage(labelimage1, path1)
                # pathedge = os.path.join(valpath,
                #                     f"ATM_{predList[i]['name']}_predge.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                # pathedge1 = os.path.join(valpath,
                #                      f"ATM_{predList[i]['name']}_predge1.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                # # sitk.WriteImage(edgeimage, pathedge)
                # # sitk.WriteImage(edgeimage1, pathedge1)
                # pathedgeadd = os.path.join(valpath,
                #                         f"ATM_{predList[i]['name']}_predgeadd.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                # pathedgeadd1 = os.path.join(valpath,
                #                          f"ATM_{predList[i]['name']}_predgeadd1.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                # sitk.WriteImage(labeladdedgeImage, pathedgeadd)
                # sitk.WriteImage(labeladdedgeImage1, pathedgeadd1)

                #计算metrics
                if os.path.exists(inputLabelPath):
                    if datasetflag=="ATM22":
                        path = os.path.join(inputLabelPath,f"ATM_{predList[i]['name']}_0000_label.nii.gz")
                        parsepath = os.path.join(skepath, f"ATM_{predList[i]['name']}_0000__parse.nii.gz")
                    elif datasetflag=="Aeropath":
                        path = os.path.join(inputLabelPath, f"{predList[i]['name']}_CT_HR_labels.nii.gz")
                        parsepath = os.path.join(skepath, f"{predList[i]['name']}_CT_HR__parse.nii.gz")
                    elif datasetflag=='Parse22':
                        path = os.path.join(inputLabelPath, f"{predList[i]['name']}_label.nii.gz")
                        parsepath = os.path.join(skepath, f"{predList[i]['name']}_parse.nii.gz")

                    # path = os.path.join(inputLabelPath,f"ATM_{predList[i]['name']}_0000_label.nii.gz")
                    # path = os.path.join(inputLabelPath, f"{predList[i]['name']}_CT_HR_labels.nii.gz")
                    inputLabelImage = sitk.ReadImage(path)
                    inputLabelImage_np = sitk.GetArrayFromImage(inputLabelImage)
                    inputLabelSke_np = skeletonize(inputLabelImage_np)

                    # parsepath = os.path.join(skepath,f"ATM_{predList[i]['name']}_0000__parse.nii.gz")
                    # parsepath = os.path.join(skepath, f"{predList[i]['name']}_CT_HR__parse.nii.gz")
                    parseImage = sitk.ReadImage(parsepath)
                    parse_np = sitk.GetArrayFromImage(parseImage)

                    td = atm22.tree_length_calculation(labelnp,inputLabelSke_np)
                    _,_,bd = atm22.branch_detected_calculation(labelnp,parse_np,inputLabelSke_np)
                    # bd=0
                    dcs = atm22.dice_coefficient_score_calculation(labelnp,inputLabelImage_np)
                    pre = atm22.precision_calculation(labelnp,inputLabelImage_np)
                    sen = atm22.sensitivity_calculation(labelnp,inputLabelImage_np)
                    spe = atm22.specificity_calculation(labelnp,inputLabelImage_np)

                    totalMetrics = 0.25 * td + 0.25 * bd + 0.25 * dcs + 0.25 * pre
                    with open(errpath, "a") as csvout:
                        writer = csv.writer(csvout)
                        row = [predList[i]['name'],  totalMetrics, td,  bd, dcs,  pre,  sen,  spe]
                        writer.writerow(row)
                        csvout.close()
                    metricsList[0] = metricsList[0] + totalMetrics
                    metricsList[1] = metricsList[1] + td
                    metricsList[2] = metricsList[2] + bd
                    metricsList[3] = metricsList[3] + dcs
                    metricsList[4] = metricsList[4] + pre
                    metricsList[5] = metricsList[5] + sen
                    metricsList[6] = metricsList[6] + spe

                else:
                    print('input label path is not existing')

            else:
                print(f"未完成完整图像")

        if numoflist >0:
            with open(errpath, 'a') as csvout:
                writer = csv.writer(csvout)
                row = [item / numoflist for item in metricsList]
                row.insert(0,'meanMetrics: ')
                writer.writerow(row)
                csvout.close()
        else:
            print('预测结果列表为空')
    else:
        print("预测子区域结果未存入列表")

def combineImage_featuremap(predList,valpath,inputLabelPath,skepath,epoch=0,datasetflag="",featureNum=3):
    """

    :param predList:
    列表元素：   tempDics = {"name": name, "info": labelInfo, "sub": subVolumList}
    info:
    info=[]
            info.append(labels.GetSize())
            info.append(labels.GetOrigin())
            info.append(labels.GetSpacing())
            info.append(labels.GetDirection())
    subVolumList = [siglabel, subPosition, prelabeledges]
    subPosition为裁剪区域在原图中的位置，原图为numpy数组类型zyx subposition 列表内为张量
    :return:
    """
    if isinstance(predList, list):
        if not os.path.exists(valpath):
            os.makedirs(valpath)
        errpath = os.path.join(valpath,"valmetrics.csv")
        with open(errpath, "a") as csvout:
            writer = csv.writer(csvout)
            writer.writerow(["epoch: ", epoch])
            row = ["index", 'totalMetrics', 'TD', 'BD', 'DCS', "accuracy", 'sensitive', 'specificity']
            writer.writerow(row)
            csvout.close()

        metricsList = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        numoflist = len(predList)

        for i in range(len(predList)):

            infolist = predList[i]['info']
            labelnp = np.zeros(infolist[0][::-1])
            edgenp = np.zeros(infolist[0][::-1])
            if featureNum>3:
                mainnp = np.zeros(infolist[0][::-1])
                dec0np = np.zeros(infolist[0][::-1])
                catfnp = np.zeros(infolist[0][::-1])
            tagnp = np.zeros(infolist[0][::-1])
            subVolum = predList[i]['sub']

            for j in range(int(len(subVolum)/featureNum)):
                ip = subVolum[j*featureNum+1]
                labelnp[ip[0]:ip[1],ip[2]:ip[3],ip[4]:ip[5]] += subVolum[j*featureNum]
                edgenp[ip[0]:ip[1],ip[2]:ip[3],ip[4]:ip[5]] += subVolum[j*featureNum+2]
                if featureNum>3:
                    mainnp[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]] += subVolum[j * featureNum+3]
                    dec0np[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]] += subVolum[j * featureNum + 4]
                    catfnp[ip[0]:ip[1], ip[2]:ip[3], ip[4]:ip[5]] += subVolum[j * featureNum + 5]

                tagnp[ip[0]:ip[1],ip[2]:ip[3],ip[4]:ip[5]] +=1
            if np.all(tagnp != 0):
                labelnp1 = labelnp / tagnp
                edgenp1 = edgenp / tagnp
                labeladdedgenp1 = labelnp1 + edgenp1
                # return labelnp
                labelnp = (labelnp1 > 0.5).astype(dtype='uint8')
                edgenp = (edgenp1 > 0.5).astype(dtype='uint8')
                labeladdedgenp = (labeladdedgenp1 > 0.5).astype(dtype='uint8')
                # labelnp = (labelnp > 0.8).astype(dtype='uint8')

                # 提取最大连通分量
                labelnp = maxConnect(labelnp)

                labelimage = sitk.GetImageFromArray(labelnp)
                labelimage1 = sitk.GetImageFromArray(labelnp1)
                edgeimage = sitk.GetImageFromArray(edgenp)
                edgeimage1 = sitk.GetImageFromArray(edgenp1)
                labeladdedgeImage = sitk.GetImageFromArray(labeladdedgenp)
                labeladdedgeImage1 = sitk.GetImageFromArray(labeladdedgenp1)
                # a = [t.item() for t in infolist[1]]
                labelimage.SetOrigin([t.item() for t in infolist[1]])
                labelimage.SetSpacing([t.item() for t in infolist[2]])
                labelimage.SetDirection([t.item() for t in infolist[3]])
                labelimage1.CopyInformation(labelimage)
                edgeimage.CopyInformation(labelimage)
                edgeimage1.CopyInformation(labelimage)
                labeladdedgeImage.CopyInformation(labelimage)
                labeladdedgeImage1.CopyInformation(labelimage)

                if not os.path.exists(valpath):
                    os.makedirs(valpath)
                path = os.path.join(valpath,f"ATM_{predList[i]['name']}_pre.nii.gz")#'./result/val/'+predList[i][name] +"nii.gz"
                path1 = os.path.join(valpath,
                                    f"ATM_{predList[i]['name']}_pre1.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                sitk.WriteImage(labelimage,path)
                sitk.WriteImage(labelimage1, path1)
                pathedge = os.path.join(valpath,
                                    f"ATM_{predList[i]['name']}_predge.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                pathedge1 = os.path.join(valpath,
                                     f"ATM_{predList[i]['name']}_predge1.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                # sitk.WriteImage(edgeimage, pathedge)
                # sitk.WriteImage(edgeimage1, pathedge1)
                pathedgeadd = os.path.join(valpath,
                                        f"ATM_{predList[i]['name']}_predgeadd.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                pathedgeadd1 = os.path.join(valpath,
                                         f"ATM_{predList[i]['name']}_predgeadd1.nii.gz")  # './result/val/'+predList[i][name] +"nii.gz"
                # sitk.WriteImage(labeladdedgeImage, pathedgeadd)
                # sitk.WriteImage(labeladdedgeImage1, pathedgeadd1)
                if featureNum>3:
                    mainnp = mainnp / tagnp
                    dec0np = dec0np / tagnp
                    catfnp = catfnp /tagnp
                    mainImage = sitk.GetImageFromArray(mainnp)
                    dec0Image = sitk.GetImageFromArray(dec0np)
                    catfImage = sitk.GetImageFromArray(catfnp)
                    mainImage.CopyInformation(labelimage)
                    dec0Image.CopyInformation(labelimage)
                    catfImage.CopyInformation(labelimage)
                    pathmain = os.path.join(valpath,f"ATM_{predList[i]['name']}_main.nii.gz")
                    pathdec0 = os.path.join(valpath,f"ATM_{predList[i]['name']}_dec0.nii.gz")
                    pathdcatf = os.path.join(valpath,f"ATM_{predList[i]['name']}_catf.nii.gz")
                    sitk.WriteImage(mainImage,pathmain)
                    sitk.WriteImage(dec0Image,pathdec0)
                    sitk.WriteImage(catfImage,pathdcatf)

                #计算metrics
                if os.path.exists(inputLabelPath):
                    if datasetflag=="ATM22":
                        path = os.path.join(inputLabelPath,f"ATM_{predList[i]['name']}_0000_label.nii.gz")
                        parsepath = os.path.join(skepath, f"ATM_{predList[i]['name']}_0000__parse.nii.gz")
                    elif datasetflag=="Aeropath":
                        path = os.path.join(inputLabelPath, f"{predList[i]['name']}_CT_HR_labels.nii.gz")
                        parsepath = os.path.join(skepath, f"{predList[i]['name']}_CT_HR__parse.nii.gz")

                    # path = os.path.join(inputLabelPath,f"ATM_{predList[i]['name']}_0000_label.nii.gz")
                    # path = os.path.join(inputLabelPath, f"{predList[i]['name']}_CT_HR_labels.nii.gz")
                    inputLabelImage = sitk.ReadImage(path)
                    inputLabelImage_np = sitk.GetArrayFromImage(inputLabelImage)
                    inputLabelSke_np = skeletonize(inputLabelImage_np)

                    # parsepath = os.path.join(skepath,f"ATM_{predList[i]['name']}_0000__parse.nii.gz")
                    # parsepath = os.path.join(skepath, f"{predList[i]['name']}_CT_HR__parse.nii.gz")
                    parseImage = sitk.ReadImage(parsepath)
                    parse_np = sitk.GetArrayFromImage(parseImage)

                    td = atm22.tree_length_calculation(labelnp,inputLabelSke_np)
                    _,_,bd = atm22.branch_detected_calculation(labelnp,parse_np,inputLabelSke_np)
                    # bd=0
                    dcs = atm22.dice_coefficient_score_calculation(labelnp,inputLabelImage_np)
                    pre = atm22.precision_calculation(labelnp,inputLabelImage_np)
                    sen = atm22.sensitivity_calculation(labelnp,inputLabelImage_np)
                    spe = atm22.specificity_calculation(labelnp,inputLabelImage_np)

                    totalMetrics = 0.25 * td + 0.25 * bd + 0.25 * dcs + 0.25 * pre
                    with open(errpath, "a") as csvout:
                        writer = csv.writer(csvout)
                        row = [predList[i]['name'],  totalMetrics, td,  bd, dcs,  pre,  sen,  spe]
                        writer.writerow(row)
                        csvout.close()
                    metricsList[0] = metricsList[0] + totalMetrics
                    metricsList[1] = metricsList[1] + td
                    metricsList[2] = metricsList[2] + bd
                    metricsList[3] = metricsList[3] + dcs
                    metricsList[4] = metricsList[4] + pre
                    metricsList[5] = metricsList[5] + sen
                    metricsList[6] = metricsList[6] + spe

                else:
                    print('input label path is not existing')

            else:
                print(f"未完成完整图像")

        if numoflist >0:
            with open(errpath, 'a') as csvout:
                writer = csv.writer(csvout)
                row = [item / numoflist for item in metricsList]
                row.insert(0,'meanMetrics: ')
                writer.writerow(row)
                csvout.close()
        else:
            print('预测结果列表为空')
    else:
        print("预测子区域结果未存入列表")
def maxConnect(preLabel):
    # 闭运算
    # structEle = cube(2)
    # preLabel = closing(preLabel, structEle)
    # 对分割结果进行标记，为每个连通分量分配一个不同的标签
    labels = measure.label(preLabel, connectivity=preLabel.ndim)

    # 计算每个连通分量的属性，包括面积（对于2D是像素数，对于3D是体素数）
    props = measure.regionprops(labels)

    # 获取图像的中心坐标
    center = np.array(preLabel.shape) / 2.0

    # 计算最短轴的长度
    shortest_axis_length = min(preLabel.shape)

    # 设置阈值为最短轴长度的四分之一
    threshold = shortest_axis_length / 2.0

    # 初始化最大面积和对应的标签
    max_area = 0
    largest_component_label = None

    # 遍历每个连通分量
    for prop in props:
        # 计算当前连通分量质心与图像中心的距离
        centroid = prop.centroid
        distance = np.linalg.norm(center - centroid)

        # 检查是否在距离阈值内并且面积是否最大
        if distance <= threshold and prop.area > max_area:
            # if distance <= threshold and prop.area > max_area:
            max_area = prop.area
            largest_component_label = prop.label

    # 如果没有找到符合条件的连通分量，则返回原始数组
    if largest_component_label is None:
        return preLabel
    # # 找到面积最大的连通分量的标签（假设面积越大代表连通分量越大）
    # if props:  # 确保有连通分量存在
    #     # largest_component_label = max(props, key=lambda x: x.area)['label']
    #     #第二大连通分量
    #     # 按面积排序连通分量
    #     sorted_regions = sorted(props, key=lambda region: region.area, reverse=True)
    #     largest_component_label = sorted_regions[1].label

    # else:
    #     return preLabel  # 如果没有连通分量，返回原数组

    # 仅保留最大连通分量
    largest_component = (labels == largest_component_label)
    # if np.any(largest_component):
    #     print("1")
    # else:
    #     print("0")

    return largest_component.astype(dtype="uint8")

def calLoss(totallossList,lossnum =4, valpath=None,epoch = 0):
    """

    :param
    totallossList
    tempDics = {"name":name[0], "loss":sloss, "sampleNum":1}
    sloss为所有损失的列表
    :return:
    """
    if isinstance(totallossList, list):
        print("val epoch ")
        if not os.path.exists(valpath):
            os.makedirs(valpath)
        errpath = os.path.join(valpath,"valerror.csv")

        allLossVal = [0.0 for _ in range(lossnum)]
        tempLossVal = [0.0 for _ in range(lossnum)]

        with open(errpath, "a") as csvout:
            writer = csv.writer(csvout)
            writer.writerow(["epoch: ", epoch])
            row = ["index", 'totalloss', 'celoss', 'diceloss', 'edgeloss']
            writer.writerow(row)
            csvout.close()

        for i in range(len(totallossList)):
            loss = totallossList[i]["loss"]
            if loss:
                for j in range(lossnum):
                    tempLossVal[j] = loss[j].item()/totallossList[i]["sampleNum"]
                print(f"{totallossList[i]['name']}: totalloss {tempLossVal}")
                with open(errpath, "a") as csvout:
                    writer = csv.writer(csvout)
                    writer.writerow([totallossList[i]['name'], tempLossVal])
                    csvout.close()
            else:
                print(f"{totallossList[i]['name']}未获得损失")
                break

            for j in range(lossnum):
                allLossVal[j] = allLossVal[j] + tempLossVal[j]

        for i in range(lossnum):
            allLossVal[i] = allLossVal[i] /len(totallossList)

        print(f"val loss:{allLossVal}")
        with open(errpath, "a") as csvout:
            writer = csv.writer(csvout)
            writer.writerow(['valloss', allLossVal])
            csvout.close()

        return allLossVal

        # if lossnum == 4:
        #     with open(errpath, "a") as csvout:
        #         writer = csv.writer(csvout)
        #         writer.writerow(["epoch: ", epoch])
        #         row = ["index", 'totalloss', 'celoss', 'diceloss', 'edgeloss']
        #         writer.writerow(row)
        #         csvout.close()
        #     allLoss=[0.0, 0.0, 0.0, 0.0]
        #     for i in range(len(totallossList)):
        #
        #         loss = totallossList[i]["loss"]
        #
        #         tloss = 0.0
        #         ce = 0.0
        #         dice = 0.0
        #         edge = 0.0
        #
        #         num = int(len(loss)/lossnum)
        #         if num > 0:
        #             for j in range(num):
        #                 tloss += loss[j*lossnum].item()
        #                 ce += loss[j*lossnum+1].item()
        #                 dice += loss[j * lossnum + 2].item()
        #                 edge += loss[j * lossnum + 3].item()
        #             tloss /= num
        #             ce /= num
        #             dice /= num
        #             edge /= num
        #             print(f"{totallossList[i]['name']}: totalloss {tloss}、celoss {ce}、diceloss {dice}、edgeloss{edge}")
        #             with open(errpath, "a") as csvout:
        #                 writer = csv.writer(csvout)
        #                 writer.writerow([totallossList[i]['name'],  tloss,  ce,  dice,  edge])
        #                 csvout.close()
        #         else:
        #             print(f"{totallossList[i]['name']}未获得损失")
        #         allLoss[0] += tloss
        #         allLoss[1] += ce
        #         allLoss[2] += dice
        #         allLoss[3] += edge
        #
        #     allLoss[0] /= len(totallossList)
        #     allLoss[1] /= len(totallossList)
        #     allLoss[2] /= len(totallossList)
        #     allLoss[3] /= len(totallossList)
        #     print(f"val loss:{allLoss}")
        #     with open(errpath, "a") as csvout:
        #         writer = csv.writer(csvout)
        #         writer.writerow(['valloss', allLoss])
        #         csvout.close()
        #
        #     return allLoss
        # elif lossnum == 3:
        #     with open(errpath, "a") as csvout:
        #         writer = csv.writer(csvout)
        #         row = ["index", 'totalloss', 'celoss', 'diceloss']
        #         writer.writerow(row)
        #         csvout.close()
        #     allLoss=[0.0, 0.0, 0.0]
        #     for i in range(len(totallossList)):
        #
        #         loss = totallossList[i]["loss"]
        #
        #         tloss = 0.0
        #         ce = 0.0
        #         dice = 0.0
        #         # edge = 0.0
        #
        #         num = int(len(loss)/lossnum)
        #         if num > 0:
        #             for j in range(num):
        #                 tloss += loss[j*lossnum].item()
        #                 ce += loss[j*lossnum+1].item()
        #                 dice += loss[j * lossnum + 2].item()
        #                 # edge += loss[j * lossnum + 3].item()
        #             tloss /= num
        #             ce /= num
        #             dice /= num
        #             # edge /= num
        #             print(f"{totallossList[i]['name']}: totalloss {tloss}、celoss {ce}、diceloss {dice}")
        #             with open(errpath, "a") as csvout:
        #                 writer = csv.writer(csvout)
        #                 writer.writerow([totallossList[i]['name'], tloss, ce, dice])
        #                 csvout.close()
        #         else:
        #             print(f"{totallossList[i]['name']}未获得损失")
        #         allLoss[0] += tloss
        #         allLoss[1] += ce
        #         allLoss[2] += dice
        #         # allLoss[3] += edge
        #
        #     allLoss[0] /= len(totallossList)
        #     allLoss[1] /= len(totallossList)
        #     allLoss[2] /= len(totallossList)
        #     # allLoss[3] /= len(totallossList)
        #     print(f"val loss:{allLoss}")
        #     with open(errpath, "a") as csvout:
        #         writer = csv.writer(csvout)
        #         writer.writerow(['valloss', allLoss])
        #         csvout.close()
        #
        #     return allLoss

    else:
        print("预测子区域的损失结果未存入列表")

if __name__ == "__main__":
    # errpath = os.path.join("./result", "error.csv")
    # a = [0,0,0]
    # with open(errpath, "a") as csvout:
    #     writer = csv.writer(csvout)
    #     writer.writerow(['name', a])
    #     csvout.close()
    inputLabelPath =''
    if os.path.exists(inputLabelPath):
        print("cunzai")
    else:
        print("bucunzai")