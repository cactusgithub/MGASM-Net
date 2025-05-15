import argparse
import gc
import os.path
import sys
# sys.path.append('./')

from models.Mamba3D_gatedShape import Mamba3DShape
from models.Mamba3D_gatedShape_Bi import Mamba3DShape as Bi
from models.vnetShape import VNet as vnets
from data import AirwayData
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import weights_init, combineImage, calLoss
from tqdm import tqdm
import torch
from loss import totalLoss
import csv
import math
from torch.cuda.amp import autocast,GradScaler
import random
from trainer import trainNet,valNet,valNetCOPD,valNetFeature

# import pdb
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--endepochs",default=150,type=int,help='number of total running epoch ')
    parser.add_argument("--batchsize",default=1,type=int,help='number of batch size ')

    parser.add_argument('--datapath',default='./Data/Data',type=str,help='path of data')
    parser.add_argument('--resumepart',default=0,type=int,help='resume part parameters')
    parser.add_argument("--optim_name",default='adam',type=str,help='optimizer')

    parser.add_argument("--poly_exp", default=1.0, type=float, help="多项式衰减的指数0.5")
    parser.add_argument("--optim_lr",default=0.0001,type=float,help="learning rate0.001")

    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--valFreq", default=10, type=float, help="validation frequence")

    parser.add_argument("--randomCrop", default=True, type=bool, help="sample set random crop")
    parser.add_argument("--BLoss", default=False, type=bool, help="use boudary loss, the number of category is greater than or equal to 2 ")
    parser.add_argument("--Sampling", default=False, type=bool, help="Sampling sample set ")
    parser.add_argument("--edge", default=True, type=bool, help="add edge module ")
    # 采样策略开关
    parser.add_argument("--randomCropSampling", default=True, type=bool, help="using data sampling strategy to sampling the data cropped randomly ")

    # 关于数据集标志和label、ske路径的配置
    # parser.add_argument('--labelpath', default='./Data/val(unresampling)/label', type=str, help='path of label file')
    # parser.add_argument('--skepath', default='./Data/val(unresampling)/ske', type=str, help='path of ske file for evaluation')
    # parser.add_argument('--labelpath', default='./Data/Aeropath/label', type=str, help='path of label file')
    # parser.add_argument('--skepath', default='./Data/Aeropath/ske', type=str,
    #                     help='path of ske file for evaluation')

    # parser.add_argument('--dataset_val', default='BAS', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_tra', default='BAS', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_test', default='BAS', type=str, help='flag of dataset')
    # parser.add_argument('--labelpath', default='./Data/BAS/val/label', type=str, help='path of label file')
    # parser.add_argument('--skepath', default='./Data/BAS/val/ske', type=str,
    #                     help='path of ske file for evaluation')
    # -----------------------------------------------------------------------------

    parser.add_argument('--cropsize', default=(128, 128, 128), nargs="*", type=int, help='size of crop input image')
    parser.add_argument('--stride', default=(127, 127, 127), nargs="*", type=int,
                        help='stride of crop input image in val')
    # parser.add_argument("--samPointNum", default=5, type=int, help='number of sampling skeleton points ')
    # parser.add_argument("--samPointDis", default=50, type=int, help='100distance of sampling skeleton points ')
    parser.add_argument("--au", default=False, type=bool, help="Whether to perform image enhancement")

    # ATM22 数据集
    parser.add_argument('--dataset_val', default='ATM22', type=str, help='flag of dataset')
    parser.add_argument('--dataset_tra', default='ATM22', type=str, help='flag of dataset')
    parser.add_argument('--dataset_test', default='ATM22', type=str, help='flag of dataset')
    parser.add_argument('--labelpath', default='./Data/Data/val(unresampling)/label', type=str, help='path of label file')
    parser.add_argument('--skepath', default='./Data/Data/val(unresampling)/ske', type=str,
                        help='path of ske file for evaluation')
    parser.add_argument("--samPointNum", default=2, type=int, help='number of sampling skeleton points ')
    parser.add_argument("--samPointDis", default=100, type=int, help='distance of sampling skeleton points ')

    # parser.add_argument('--testpath', default='./unresampling/result(sdec)/test', type=str,
    #                     help='prediction result of test')

    # train in different parament
    # parser.add_argument("--logpath", default='./unresampling/result(i12896128,sampling,6d,ske,bod,edice)/threshold0.8', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(i12896128,sampling,6d,ske,bod,edice)threshold0.8', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode',default='val',type=str,help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(i12896128,sampling,6d,ske,bod,edice)/val/threshold0.8', type=str, help='prediction result of validation')

    #————————————train——————————————————————————————————————————————
    # shape guide——sdecoder 六向
    parser.add_argument("--logpath", default='./unresampling/result(sdec32_2)', type=str, help="record loss")
    parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32_2)', type=str, help='save checkpoint path')

    parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful onrestarts)')
    parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    parser.add_argument("--resume", default='./unresampling/result(sdec32_2)/060.ckpt', type=str, help='the latest path for checkpoint')
    parser.add_argument('--valpath', default='./unresampling/result(sdec32_2)/val', type=str, help='prediction result of validation')
    parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 双向
    # parser.add_argument("--logpath", default='./unresampling/result(sdecbi32)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdecbi32)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=21, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdecbi32)/020.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdecbi32)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='bi', type=str, help='types of trained model')

    # shape guide——sdecoder 六向randomcrop sampling 1,100
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32rs1_1)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32rs1_1)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=101, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32rs1_1)/100.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdec32rs1_1)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 六向randomcrop sampling 2,10
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32rs2d10)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32rs2d10)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=101, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32rs2d10)/100.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdec32rs2d10)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 六向 randomcrop
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32r)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32r)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=81, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32r)/080.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdec32r)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 六向randomcrop sampling 3,100
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32rs3d100)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32rs3d100)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=71, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32rs3d100)/070.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdec32rs3d100)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 六向randomcrop sampling 4,100
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32rs4d100)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32rs4d100)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdec32rs4d100)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # morphological guided experiments——————————————————————————————————————————————————————————————————————
    # parser.add_argument("--logpath", default='./unresampling/result(vnets)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(vnets)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=201, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(vnets)/200.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(vnets)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='vents', type=str, help='types of trained model')

    # shape guide——sencoder 作用于mamba输出 六向
    # parser.add_argument("--logpath", default='./unresampling/result(senc)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(senc)', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful onrestarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(senc)/val', type=str, help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')



    # —————————————————Au—————————————————————————————————————————————————————————————————————————————————————————————
    # shape guide——sdecoder
    # parser.add_argument("--logpath", default='./unresampling/result(sdecAu)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdecAu)', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdecAu)/val', type=str, help='prediction result of validation')

    # shape guide——sdecoder augmentation from 95
    # parser.add_argument("--logpath", default='./unresampling/result(sdecAu95)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdecAu95)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=131, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdecAu95)/130.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdecAu95)/val', type=str,
    #                     help='prediction result of validation')


    # shape guide st
    # parser.add_argument("--logpath", default='./unresampling/result(st)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(st)', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=51, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(st)/050.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(st)/val', type=str,
    #                     help='prediction result of validation')

    # val————————————————————————————————————————————————————————————————————
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='val', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda:2', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./result(space111crop128b2weight1-0.5depth1251)/300.ckpt', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./result(space111crop128b2weight1-0.5depth1251)/val', type=str,
    #                     help='prediction result of validation')

    # sdec val output all feature
    # parser.add_argument("--logpath", default='./unresampling/result(sdec)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec)', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=95, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='val', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec)/095.ckpt', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdec)/valall', type=str,
    #                     help='prediction result of validation')

    # test——————————————————————————————————
    # parser.add_argument("--logpath", default='./unresampling/result(sdec)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec)', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=95, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec)/095.ckpt', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdec)/testall', type=str,
    #                     help='prediction result of validation')

    # shape guide st
    # parser.add_argument("--logpath", default='./unresampling/result(st)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(st)', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(st)/100.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(st)/test', type=str,
    #                     help='prediction result of validation')

    # bas 数据集—————train—————————————————————————————————————————————
    # parser.add_argument('--dataset_val', default='BAS', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_tra', default='BAS', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_test', default='BAS', type=str, help='flag of dataset')
    # parser.add_argument('--labelpath', default='./Data/BAS/val/label', type=str, help='path of label file')
    # parser.add_argument('--skepath', default='./Data/BAS/val/ske', type=str,
    #                     help='path of ske file for evaluation')
    #
    #
    # parser.add_argument("--logpath", default='./unresampling/BAS/result(sdec_s5)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/BAS/result(sdec_s5)', type=str, help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/BAS/result(sdec_s5)/val1', type=str,
    #                     help='prediction result of validation')
    #

    # COPD 数据集————————test————————————————————————————————————————
    # parser.add_argument("--logpath", default='./unresampling/COPD/result(sdec)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/COPD/result(sdec)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=1, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec)/095.ckpt', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/COPD/result(sdec)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--dataset_val', default='', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_tra', default='', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_test', default='COPD', type=str, help='flag of dataset')
    # parser.add_argument('--labelpath', default='', type=str, help='path of label file')
    # parser.add_argument('--skepath', default='', type=str,
    #                     help='path of ske file for evaluation')
    #
    # # # --shape guide——sdecoder 六向
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=60, type=int, help='manual epoch number (useful on restarts)')
    # # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32)/060.ckpt', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdec32)/COPD/testCOPD60', type=str,
    #                     help='prediction result of validation')
    # # parser.add_argument('--valpath', default='./unresampling/result(sdec32)/val', type=str,
    # #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # AeroPath 数据集  test——————————————————————————————————————————————————————————————————————————
    # parser.add_argument('--dataset_val', default='Aeropath', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_tra', default='ATM22', type=str, help='flag of dataset')
    # parser.add_argument('--dataset_test', default='Aeropath', type=str, help='flag of dataset')
    # parser.add_argument('--labelpath', default='./Data/Data/Aeropath/label', type=str, help='path of label file')
    # parser.add_argument('--skepath', default='./Data/Data/Aeropath/ske', type=str,
    #                     help='path of ske file for evaluation')
    #
    # # --shape guide——sdecoder 六向
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=160, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # # parser.add_argument('--mode', default='val', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32)/160.ckpt', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdec32)/testAero160', type=str,
    #                     help='prediction result of validation')
    # # parser.add_argument('--valpath', default='./unresampling/result(sdec32)/val', type=str,
    # #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 双向
    # parser.add_argument("--logpath", default='./unresampling/result(sdecbi32)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdecbi32)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=50, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdecbi32)/050.ckpt', type=str,
    #                     help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdecbi32)/testAero', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='bi', type=str, help='types of trained model')

    # shape guide——sdecoder 六向randomcrop sampling 1,100
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32rs1_1)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32rs1_1)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=200, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32rs1_1)/200.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdec32rs1_1)/testAero200', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 六向randomcrop sampling 2,10
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32rs2d10_1)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresamplin   g/result(sdec32rs2d10_1)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=90, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32rs2d10_1)/090.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdec32rs2d10_1)/testAero90/case9', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 六向 randomcrop
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32r)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32r)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=81, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='train', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32r)/080.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--valpath', default='./unresampling/result(sdec32r)/val', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——sdecoder 六向randomcrop sampling 3,100
    # parser.add_argument("--logpath", default='./unresampling/result(sdec32rs3d100)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(sdec32rs3d100)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=60, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(sdec32rs3d100)/060.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(sdec32rs3d100)/testAero90', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')

    # shape guide——senc
    # parser.add_argument("--logpath", default='./unresampling/result(senc)', type=str, help="record loss")
    # parser.add_argument('--saveModelPath', default='./unresampling/result(senc)', type=str,
    #                     help='save checkpoint path')
    #
    # parser.add_argument("--startepochs", default=60, type=int, help='manual epoch number (useful on restarts)')
    # parser.add_argument('--mode', default='test', type=str, help='train 、val or test')
    # parser.add_argument('--cudaDevice', default='cuda', type=str, help='number of cuda device')
    # parser.add_argument("--resume", default='./unresampling/result(senc)/060.ckpt', type=str, help='the latest path for checkpoint')
    # parser.add_argument('--testpath', default='./unresampling/result(senc)/testAero60', type=str,
    #                     help='prediction result of validation')
    # parser.add_argument('--model', default='base', type=str, help='types of trained model')
    #



    args = parser.parse_args()
    seed = args.startepochs - 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device(args.cudaDevice if torch.cuda.is_available() else "cpu")
    # device = torch.device( "cpu")

    # lr = 0.001#########

    #创建模型
    if args.model == "bi":
        model = Bi()
    elif args.model == 'vents':
        model = vnets()
    else:
        model = Mamba3DShape()
    print(model)

    print('# of network parameters:', sum(param.numel() for param in model.parameters()))
    # criterion = totalLoss()#sunshihanshu
    # 损失函数 设置权重
    criterion = totalLoss(lmbda=1, epsilon=1, alpha=1)
    scaler = GradScaler()

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    lambda1 = lambda epoch: math.pow(1 - args.startepochs / args.endepochs, args.poly_exp)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1 )
    # optimizer = torch.optim.SGD(model.parameters(),lr,0.1)
    #加载权重
    if args.resume:
        checkpoints = torch.load(args.resume)
        if args.resumepart:
            model.load_state_dict(checkpoints['state_dict'],strict=False)
            print('part load Done')
        else:
            model.load_state_dict(checkpoints['state_dict'])
            print("full resume Done")
    else:
        #初始化权重
        weights_init(model, init_type='kaiming')  # weight initialization

    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)

    model.to(device)
    criterion.to(device)

    #加载数据
    if args.mode == "train":
        trainData = AirwayData(args.datapath,args.mode,randomCropFlag=args.randomCrop, cropFfactor=args.cropsize,stride=args.stride,
                               bodFlag=args.BLoss, sampleFlag=args.Sampling,
                               edgeFlag=args.edge,samPointNum=args.samPointNum,samPointDis=args.samPointDis,
                               datasetflag=args.dataset_tra,augmentationflag=args.au,randomCropSamplingFlag=args.randomCropSampling)
        trainLoader = DataLoader(trainData,args.batchsize,shuffle=True)

        valData = AirwayData(args.datapath, 'val', cropFfactor=args.cropsize,stride=args.stride,
                             bodFlag=args.BLoss,edgeFlag=args.edge,datasetflag=args.dataset_val)
        valLoader = DataLoader(valData, 1, shuffle=False)
    elif args.mode == "test":
        testData = AirwayData(args.datapath,args.mode,cropFfactor=args.cropsize,stride=args.stride,
                              bodFlag=args.BLoss,edgeFlag=args.edge,datasetflag=args.dataset_test)
        testLoader = DataLoader(testData,1,shuffle=False)

        if not os.path.exists(args.logpath):
            os.makedirs(args.logpath)
        if args.dataset_test == 'COPD':
            testLoss = valNetCOPD(testLoader, model, device, criterion, args.testpath, args.labelpath, args.skepath,
                              args.startepochs)
            print(f'epoch{1} testloss:{testLoss}')
            sys.exit()
        else:
            testLoss = valNet(testLoader, model, device, criterion, args.testpath, args.labelpath, args.skepath, args.startepochs,datasetflag=args.dataset_test)
            # testLoss = valNetFeature(testLoader, model, device, criterion, args.testpath, args.labelpath, args.skepath,
            #                   args.startepochs, datasetflag=args.dataset_test)

            print(f'epoch{1} testloss:{testLoss}')
            sys.exit()


    else:
        # valData = AirwayData(args.datapath, 'val',cropFfactor=(128,96,128),stride=(127,95,127),bodFlag=args.BLoss)
        # valLoader = DataLoader(valData, 1, shuffle=False)
        valData = AirwayData(args.datapath, 'val', cropFfactor=args.cropsize, stride=args.stride, bodFlag=args.BLoss,
                             edgeFlag=args.edge,datasetflag=args.dataset_val)
        valLoader = DataLoader(valData, 1, shuffle=False)
        if not os.path.exists(args.logpath):
            os.makedirs(args.logpath)

        # valNet(valLoader,model,device,criterion,args.valpath)
        # sys.exit()
        valLoss = valNet(valLoader, model, device, criterion, args.valpath, args.labelpath, args.skepath,
                          args.startepochs,datasetflag=args.dataset_val)
        print(f'epoch{1} testloss:{valLoss}')
        sys.exit()

    if not os.path.exists(args.logpath):
        os.makedirs(args.logpath)
    logname = os.path.join(args.logpath, 'trainError.csv')
    with open(logname,"a") as csvout:
        writer = csv.writer(csvout)
        row = ["index", 'totalloss', 'celoss', 'diceloss', 'edgeloss']
        writer.writerow(row)
        csvout.close()


    for epoch in range(args.startepochs,args.endepochs+1):
        print(f"start {epoch} epoch")
        loss = trainNet(trainLoader,model,criterion,optimizer,scaler,device)

        #保存checkpoint

        if epoch%args.valFreq == 0:
            state_dict = model.state_dict()
            torch.save({"state_dict":state_dict,"args":args},os.path.join(args.saveModelPath,'%03d.ckpt' % epoch))
            print(f"epoch {epoch}: 保存checkpoint {epoch}.ckpt")
        # else:
        #     torch.save({"state_dict": state_dict, "args": args}, os.path.join(args.saveModelPath, 'lastest.ckpt'))
        #     print(f"epoch {epoch}: 保存checkpoint lastest.ckpt")
        # 保存loss

        with open(logname, 'a') as csvout:
            writer = csv.writer(csvout)
            writer.writerow([epoch,loss])
            # writer.writerow(loss)
            csvout.close()

        if scheduler is not None:
            scheduler.step()

        # print(f'epoch{epoch} trainloss:{loss}')

        if (epoch % args.valFreq == 0) or (epoch == args.startepochs):
            valLoss = valNet(valLoader, model, device, criterion, args.valpath,args.labelpath,args.skepath,epoch,datasetflag=args.dataset_val)
            # print(f'epoch{epoch} valloss:{valLoss}')

    torch.cuda.empty_cache()
    gc.collect()




