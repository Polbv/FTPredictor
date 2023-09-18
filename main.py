import math
from utils import *
import os
import os.path as osp
import random
from torch.utils.data import DataLoader
from datasets import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



        ##REGRESSION FOR SKELETON POINTS IN FT SHOT##
# trainset=FTlineardataset('data/predictions_with_linear_labels/json/train',"data/predictions/json/train",transform=True)
# # testset=FTlineardataset('data/predictions_with_linear_labels/json/test','data/predictions/json/test',transform=True)
# validationset=FTlineardataset('data/predictions_with_linear_labels/json/validation','data/predictions/json/validation',transform=True)

# trainloader=DataLoader(trainset,batch_size=5,shuffle=True)
# # testloader=DataLoader(testset,batch_size=1,shuffle=True)
# validationloader=DataLoader(validationset,batch_size=5,shuffle=True)

        ##REGRESSION USING ENGINEERED FEATURES##
# path to dataset folders
testfolder='data/inference_annotations/test'
trainfolder='data/inference_annotations/train'
validationfolder='data/inference_annotations/validation'

trainset=FTdatasetparams(trainfolder,transform=True)
testset=FTlineardataset(testfolder,transform=True)
validationset=FTdatasetparams(validationfolder,transform=True)

trainloader=DataLoader(trainset,batch_size=5,shuffle=True)
testloader=DataLoader(testset,batch_size=1,shuffle=True)
validationloader=DataLoader(validationset,batch_size=5,shuffle=True)

        ##TRAINING OR TESTING##
TRAIN=1

#LSTM PARAMETERS
num_classes=1
num_hidden=16
num_layers=1
input_size=7
batchsize=8
lr=0.0001
num_epochs=4000


        ##MODEL DEFINITION##

model=LSTMClassifier(input_size,num_hidden,num_layers,num_classes)
linear_model=LSTMlinearClassifier(input_size,num_hidden,num_layers,num_classes)
lossfn=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

#TRAINING
if TRAIN:
        train_model(linear_model,trainloader,validationloader,lossfn,optimizer,num_epochs=num_epochs)

##TESTING
else:
        try: 
                #replace with the path to the checkpoint
                ckptpath='results/Train#005/linear_model.pth'
                checkpoint=torch.load(ckptpath)
                linear_model.load_state_dict(checkpoint)
        except: pass

        linear_model.eval()

        #testing on the validation set
        results=[]
        for annotation in os.listdir('data/predictions_with_linear_labels/json/validation'):
                path=osp.join('data/predictions_with_linear_labels/json/validation',annotation)
                #videopath=''
                print (path)
                p,l=test(path,linear_model,fullpose=False)
                results.append([p,l])
        n=0
        correct=0
        incorrect=0
        print (results)
        for p,l in results:
                n+=1
                if p==l:
                        correct+=1
                else :
                        incorrect+=1
        print ('got',correct, 'correct predictions out of',n,'samples')
        print ('accuracy:',correct/n)




