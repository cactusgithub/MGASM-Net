from torch.cuda.amp import autocast
import torch
from tqdm import tqdm
from utils import combineImage, calLoss

from  torch.profiler import profiler,record_function


# def trainNet(trainLoader,model,criterion,optimizer,scaler,device):
#     model.train()
#
#     allLoss=[]
#
#     with profiler.profile(
#             activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
#             # record_shapes=True,
#     ) as prof:
#         # prof.start()
#         with record_function("model_total"):
#             for i,(imgsarr,labelsarr,labelEdgearr) in enumerate(tqdm(trainLoader)):
#
#                 with record_function('model_inference1'):
#                     datanum = len(imgsarr)
#                     for j in range(datanum):
#                         imgs = imgsarr[j]
#                         labels = labelsarr[j]
#                         labelEdge = labelEdgearr[j]
#                         optimizer.zero_grad()
#                         imgs,labels,labelEdge = imgs.to(device),labels.to(device),labelEdge.to(device)
#                         # labelEdgeDisMap = labelEdgeDisMap.to(device)
#                         with record_function('model_inference2'):
#                             with autocast():
#                                 # prelabels,siglabel,prelabelEdge = model(imgs)
#                                 prelabels, siglabel, prelabelEdge, prelabelEdges = model(imgs)
#
#                                 with record_function('model_inference_back'):
#                                     loss, sloss = criterion(prelabels, siglabel, prelabelEdge, prelabelEdges, labels,
#                                                             labelEdge=labelEdge)
#
#                                 with record_function('model_inference_back1'):
#                                     scaler.scale(loss).backward()
#                                     scaler.step(optimizer)
#                                     scaler.update()
#
#                         prof.step()
#
#                         sloss.insert(0,loss)
#                         allLoss.append(sloss)
#         prof.stop()
#         print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
#     listnumber = len(allLoss)
#     lossnum = len(sloss)
#     meanloss = [0.0 for _ in range(lossnum)]
#     for dicts in allLoss:
#         for i in range(lossnum):
#             meanloss[i] = meanloss[i] + dicts[i]
#
#     for i in range(lossnum):
#         meanloss[i] = meanloss[i] / listnumber
#
#     return meanloss

# 采样多个的版本
def trainNet(trainLoader,model,criterion,optimizer,scaler,device):
    model.train()

    allLoss=[]

    for i,(imgsarr,labelsarr,labelEdgearr) in enumerate(tqdm(trainLoader)):
        datanum = len(imgsarr)
        for j in range(datanum):
            imgs = imgsarr[j]
            labels = labelsarr[j]
            labelEdge = labelEdgearr[j]
            optimizer.zero_grad()

            imgs,labels= imgs.to(device),labels.to(device)
            labelEdge = labelEdge.to(device)
            # labelEdgeDisMap = labelEdgeDisMap.to(device)
            # with autocast():
            prelabels,siglabel,prelabelEdge,prelabelEdges= model(imgs)
            loss, sloss = criterion(prelabels, siglabel, prelabelEdge,prelabelEdges,labels, labelEdge=labelEdge)



            loss.backward()
            optimizer.step()

            sloss.insert(0,loss)

            allLoss.append(sloss)

    listnumber = len(allLoss)
    lossnum = len(sloss)
    meanloss = [0.0 for _ in range(lossnum)]
    for dicts in allLoss:
        for i in range(lossnum):
            meanloss[i] = meanloss[i] + dicts[i]

    for i in range(lossnum):
        meanloss[i] = meanloss[i] / listnumber

    return meanloss




def valNet(valLoader,model,device,criterion,valpath,labelpath='./Data/val(unresampling)/label',skepath='./Data/val(unresampling)/ske',epoch = 0,datasetflag=""):
    model.eval()
    predlist=[]
    allLoss = []

    for i, (imgs,labels,labelEdge,name, labelInfo, subPosition) in enumerate(tqdm(valLoader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        labelEdge = labelEdge.to(device)
        # labelEdgeDisMap = labelEdgeDisMap.to(device)

        with torch.no_grad():

            prelabels, siglabel, prelabelEdge, prelabelEdges = model(imgs)
            loss, sloss = criterion(prelabels, siglabel, prelabelEdge, prelabelEdges,labels,labelEdge=labelEdge, labelEdgeBoundaryMap = None)

        siglabel = siglabel.squeeze(0).squeeze(0)
        prelabelEdges = prelabelEdges.squeeze(0).squeeze(0)

        siglabel = siglabel.to("cpu")
        siglabel = siglabel.numpy()
        prelabelEdges = prelabelEdges.to("cpu")
        prelabelEdges = prelabelEdges.numpy()


        sloss.insert(0, loss)
        lossnum = len(sloss)

        for i in range(lossnum):
            sloss[i] = sloss[i].to("cpu")

        #______计算整体损失值_________#
        if not allLoss:

            tempDics = {"name":name[0], "loss":sloss, "sampleNum":1}
            allLoss.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == allLoss[j]["name"]:
                    for i in range(len(sloss)):
                        allLoss[j]["loss"][i] = allLoss[j]["loss"][i]+sloss[i]

                    allLoss[j]["sampleNum"] = allLoss[j]["sampleNum"] +1
                    tag = False

                    break
            if tag:

                tempDics = {"name": name[0], "loss": sloss, "sampleNum":1}
                allLoss.append(tempDics)

        # ______保存每一个子体积的预测值_________#
        if not predlist:
            subVolumList = [siglabel, subPosition,prelabelEdges]
            tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
            predlist.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == predlist[j]['name']:

                    predlist[j]['sub'].append(siglabel)
                    predlist[j]['sub'].append(subPosition)
                    predlist[j]['sub'].append(prelabelEdges)
                    tag = False
                    break

            if tag:
                subVolumList = [siglabel, subPosition, prelabelEdges]
                tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
                predlist.append(tempDics)

        # break


    valLoss = calLoss(allLoss,lossnum=lossnum,valpath=valpath,epoch=epoch)
    combineImage(predlist, valpath,labelpath,skepath,epoch=epoch,datasetflag=datasetflag)

    return valLoss

def valNetFeature(valLoader,model,device,criterion,valpath,labelpath='./Data/val(unresampling)/label',skepath='./Data/val(unresampling)/ske',epoch = 0,datasetflag=""):
    model.eval()
    predlist=[]
    allLoss = []

    for i, (imgs,labels,labelEdge,name, labelInfo, subPosition) in enumerate(tqdm(valLoader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        labelEdge = labelEdge.to(device)
        # labelEdgeDisMap = labelEdgeDisMap.to(device)

        with torch.no_grad():
            # with autocast():
            prelabels, siglabel, prelabelEdge, prelabelEdges,mainbranchout,dec0,catf= model(imgs)
            loss, sloss = criterion(prelabels, siglabel, prelabelEdge, prelabelEdges,labels,labelEdge=labelEdge, labelEdgeBoundaryMap = None)


                # sample ={}
            # for j  in range(batchsize):
            #     sample = {'name': name, 'preimage': pred[j].to('cpu').numpy()}
            #     predlist.append(sample)
        siglabel = siglabel.squeeze(0).squeeze(0)
        prelabelEdges = prelabelEdges.squeeze(0).squeeze(0)
        mainbranchout = mainbranchout.mean(dim=1,keepdim=False).squeeze(0)
        catf = catf.mean(dim=1,keepdim=False).squeeze(0)
        dec0 = dec0.mean(dim=1,keepdim=False).squeeze(0)
        # dec1 = dec1.squeeze(0).squeeze(0)
        # dec2 = dec2.squeeze(0).squeeze(0)
        # dec3 = dec3.squeeze(0).squeeze(0)
        mainbranchout = mainbranchout.to("cpu")
        mainbranchout = mainbranchout.numpy()
        dec0 = dec0.to("cpu")
        # dec1 = dec1.to("cpu")
        # dec2 = dec2.to("cpu")
        # dec3 = dec3.to("cpu")
        dec0 = dec0.numpy()
        # dec1 = dec1.numpy()
        # dec2 = dec2.numpy()
        # dec3 = dec3.numpy()
        catf = catf.to("cpu")
        catf = catf.numpy()


        siglabel = siglabel.to("cpu")
        siglabel = siglabel.numpy()
        prelabelEdges = prelabelEdges.to("cpu")
        prelabelEdges = prelabelEdges.numpy()


        # labels = labels.squeeze(0)
        # labelnp = labels.to('cpu')
        # labelnp = labelnp.numpy()
        #
        # labelEdge = labelEdge.squeeze(0)
        # labelEdgenp = labelEdge.to('cpu')
        # labelEdgenp = labelEdgenp.numpy()

        sloss.insert(0, loss)
        lossnum = len(sloss)
        # loss = loss.to("cpu")
        for i in range(lossnum):
            sloss[i] = sloss[i].to("cpu")
        # sloss = sloss.to("cpu")
        # celoss = celoss.to("cpu")
        # diceloss = diceloss.to("cpu")
        # edgeloss = edgeloss.to("cpu")

        # sample = {"total": loss, "CE": celoss, "dice": diceloss, "Edge": edgeloss}
        # totalloss.append(sample)
        #______计算整体损失值_________#
        if not allLoss:

            tempDics = {"name":name[0], "loss":sloss, "sampleNum":1}
            allLoss.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == allLoss[j]["name"]:
                    for i in range(len(sloss)):
                        allLoss[j]["loss"][i] = allLoss[j]["loss"][i]+sloss[i]

                    allLoss[j]["sampleNum"] = allLoss[j]["sampleNum"] +1
                    tag = False

                    break
            if tag:

                tempDics = {"name": name[0], "loss": sloss, "sampleNum":1}
                allLoss.append(tempDics)

        # ______保存每一个子体积的预测值_________#
        if not predlist:
            subVolumList = [siglabel, subPosition,prelabelEdges,mainbranchout,dec0,catf]
            tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
            predlist.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == predlist[j]['name']:

                    predlist[j]['sub'].append(siglabel)
                    predlist[j]['sub'].append(subPosition)
                    predlist[j]['sub'].append(prelabelEdges)
                    predlist[j]['sub'].append(mainbranchout)
                    # predlist[j]['sub'].append(dec3)
                    # predlist[j]['sub'].append(dec2)
                    # predlist[j]['sub'].append(dec1)
                    predlist[j]['sub'].append(dec0)
                    predlist[j]['sub'].append(catf)

                    tag = False
                    break

            if tag:
                subVolumList = [siglabel, subPosition, prelabelEdges,mainbranchout,dec0,catf]
                tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
                predlist.append(tempDics)

        # break


    valLoss = calLoss(allLoss,lossnum=lossnum,valpath=valpath,epoch=epoch)
    combineImage(predlist, valpath,labelpath,skepath,epoch=epoch,datasetflag=datasetflag,featureNum=6)
    # print(f"val loss: {meanloss}")

    # return predlist
    return valLoss
def valNetCOPD(valLoader,model,device,criterion,valpath,labelpath='',skepath='',epoch = 0):
    model.eval()
    predlist=[]
    allLoss = []

    for i, (imgs,name, labelInfo, subPosition) in enumerate(tqdm(valLoader)):
        imgs = imgs.to(device)


        with torch.no_grad():
            # with autocast():
            prelabels, siglabel, prelabelEdge, prelabelEdges = model(imgs)



                # sample ={}
            # for j  in range(batchsize):
            #     sample = {'name': name, 'preimage': pred[j].to('cpu').numpy()}
            #     predlist.append(sample)
        siglabel = siglabel.squeeze(0).squeeze(0)
        prelabelEdges = prelabelEdges.squeeze(0).squeeze(0)

        siglabel = siglabel.to("cpu")
        siglabel = siglabel.numpy()
        prelabelEdges = prelabelEdges.to("cpu")
        prelabelEdges = prelabelEdges.numpy()




        #______计算整体损失值_________# 



        # ______保存每一个子体积的预测值_________#
        if not predlist:
            subVolumList = [siglabel, subPosition,prelabelEdges]
            tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
            predlist.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == predlist[j]['name']:

                    predlist[j]['sub'].append(siglabel)
                    predlist[j]['sub'].append(subPosition)
                    predlist[j]['sub'].append(prelabelEdges)
                    tag = False
                    break

            if tag:
                subVolumList = [siglabel, subPosition, prelabelEdges]
                tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
                predlist.append(tempDics)

        # break



    combineImage(predlist, valpath,labelpath,skepath,epoch=epoch)
    # print(f"val loss: {meanloss}")

    # return predlist
    return 0



# 采样多个的版本  16表示半精度版本
def trainNet16(trainLoader,model,criterion,optimizer,scaler,device):
    model.train()

    allLoss=[]

    for i,(imgsarr,labelsarr,labelEdgearr) in enumerate(tqdm(trainLoader)):
        datanum = len(imgsarr)
        for j in range(datanum):
            imgs = imgsarr[j]
            labels = labelsarr[j]
            labelEdge = labelEdgearr[j]
            optimizer.zero_grad()

            imgs,labels= imgs.to(device),labels.to(device)
            labelEdge = labelEdge.to(device)
            # labelEdgeDisMap = labelEdgeDisMap.to(device)
            with autocast():
                prelabels,siglabel,prelabelEdge,prelabelEdges= model(imgs)
                loss, sloss = criterion(prelabels, siglabel, prelabelEdge,prelabelEdges,labels, labelEdge=labelEdge)
                # loss = criterion(prelabels, prelabelEdge, labels, labelEdge)
            # print(f"start backward {i}")
            # pdb.set_trace()
            scaler.scale(loss).backward()

            # print(f"end backward {i}")
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            # print(f"data {i}")

            # sample={"total": loss, "sloss": sloss}
            sloss.insert(0,loss)
            # sample = {"total": loss, "CE": celoss, "dice": diceloss, "Edge": edgeloss}
            allLoss.append(sloss)

    listnumber = len(allLoss)
    lossnum = len(sloss)
    meanloss = [0.0 for _ in range(lossnum)]
    for dicts in allLoss:
        for i in range(lossnum):
            meanloss[i] = meanloss[i] + dicts[i]

    for i in range(lossnum):
        meanloss[i] = meanloss[i] / listnumber

    return meanloss




def valNet16(valLoader,model,device,criterion,valpath,labelpath='./Data/val(unresampling)/label',skepath='./Data/val(unresampling)/ske',epoch = 0):
    model.eval()
    predlist=[]
    allLoss = []

    for i, (imgs,labels,labelEdge,name, labelInfo, subPosition) in enumerate(tqdm(valLoader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        labelEdge = labelEdge.to(device)
        # labelEdgeDisMap = labelEdgeDisMap.to(device)

        with torch.no_grad():
            with autocast():
                prelabels, siglabel, prelabelEdge, prelabelEdges = model(imgs)
                loss, sloss = criterion(prelabels, siglabel, prelabelEdge, prelabelEdges,labels,labelEdge=labelEdge, labelEdgeBoundaryMap = None)


                # sample ={}
            # for j  in range(batchsize):
            #     sample = {'name': name, 'preimage': pred[j].to('cpu').numpy()}
            #     predlist.append(sample)
        siglabel = siglabel.squeeze(0).squeeze(0)
        prelabelEdges = prelabelEdges.squeeze(0).squeeze(0)

        siglabel = siglabel.to("cpu")
        siglabel = siglabel.numpy()
        prelabelEdges = prelabelEdges.to("cpu")
        prelabelEdges = prelabelEdges.numpy()

        # labels = labels.squeeze(0)
        # labelnp = labels.to('cpu')
        # labelnp = labelnp.numpy()
        #
        # labelEdge = labelEdge.squeeze(0)
        # labelEdgenp = labelEdge.to('cpu')
        # labelEdgenp = labelEdgenp.numpy()

        sloss.insert(0, loss)
        lossnum = len(sloss)
        # loss = loss.to("cpu")
        for i in range(lossnum):
            sloss[i] = sloss[i].to("cpu")
        # sloss = sloss.to("cpu")
        # celoss = celoss.to("cpu")
        # diceloss = diceloss.to("cpu")
        # edgeloss = edgeloss.to("cpu")

        # sample = {"total": loss, "CE": celoss, "dice": diceloss, "Edge": edgeloss}
        # totalloss.append(sample)
        #______计算整体损失值_________#
        if not allLoss:

            tempDics = {"name":name[0], "loss":sloss, "sampleNum":1}
            allLoss.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == allLoss[j]["name"]:
                    for i in range(len(sloss)):
                        allLoss[j]["loss"][i] = allLoss[j]["loss"][i]+sloss[i]

                    allLoss[j]["sampleNum"] = allLoss[j]["sampleNum"] +1
                    tag = False

                    break
            if tag:

                tempDics = {"name": name[0], "loss": sloss, "sampleNum":1}
                allLoss.append(tempDics)

        # ______保存每一个子体积的预测值_________#
        if not predlist:
            subVolumList = [siglabel, subPosition,prelabelEdges]
            tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
            predlist.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == predlist[j]['name']:

                    predlist[j]['sub'].append(siglabel)
                    predlist[j]['sub'].append(subPosition)
                    predlist[j]['sub'].append(prelabelEdges)
                    tag = False
                    break

            if tag:
                subVolumList = [siglabel, subPosition, prelabelEdges]
                tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
                predlist.append(tempDics)

        # break


    valLoss = calLoss(allLoss,lossnum=lossnum,valpath=valpath,epoch=epoch)
    combineImage(predlist, valpath,labelpath,skepath,epoch=epoch)
    # print(f"val loss: {meanloss}")

    # return predlist
    return valLoss


def valNetCOPD16(valLoader,model,device,criterion,valpath,labelpath='',skepath='',epoch = 0):
    model.eval()
    predlist=[]
    allLoss = []

    for i, (imgs,name, labelInfo, subPosition) in enumerate(tqdm(valLoader)):
        imgs = imgs.to(device)


        with torch.no_grad():
            with autocast():
                prelabels, siglabel, prelabelEdge, prelabelEdges = model(imgs)



                # sample ={}
            # for j  in range(batchsize):
            #     sample = {'name': name, 'preimage': pred[j].to('cpu').numpy()}
            #     predlist.append(sample)
        siglabel = siglabel.squeeze(0).squeeze(0)
        prelabelEdges = prelabelEdges.squeeze(0).squeeze(0)

        siglabel = siglabel.to("cpu")
        siglabel = siglabel.numpy()
        prelabelEdges = prelabelEdges.to("cpu")
        prelabelEdges = prelabelEdges.numpy()




        #______计算整体损失值_________#



        # ______保存每一个子体积的预测值_________#
        if not predlist:
            subVolumList = [siglabel, subPosition,prelabelEdges]
            tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
            predlist.append(tempDics)
        else:
            tag = True
            for j in range(len(predlist)):
                if name[0] == predlist[j]['name']:

                    predlist[j]['sub'].append(siglabel)
                    predlist[j]['sub'].append(subPosition)
                    predlist[j]['sub'].append(prelabelEdges)
                    tag = False
                    break

            if tag:
                subVolumList = [siglabel, subPosition, prelabelEdges]
                tempDics = {"name": name[0], "info": labelInfo, "sub": subVolumList}
                predlist.append(tempDics)

        # break



    combineImage(predlist, valpath,labelpath,skepath,epoch=epoch)
    # print(f"val loss: {meanloss}")

    # return predlist
    return 0
