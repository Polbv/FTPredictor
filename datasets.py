import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
from utils import get_kp_timeseries, get_bbox,normalize,yinverter,chopper,get_labels,zero_padding,kp_merger,zero_pad_features
from FT import FTdescriptor
import os.path as osp
import os
import tensorflow as tf
import cv2
class FTdataset(torch.utils.data.Dataset):
     #dataset class for binary labels using full pose array



    def __init__(self, annotations_folder,vid_dir, transform=True, target_transform=None):
        
        dataDir=vid_dir
        self.annfolder=annotations_folder
        self.vid_dir=vid_dir
        self.transform = transform
        



    def __len__(self):
        list=os.listdir(self.annfolder)
        return len(list)

    def __getitem__(self, idx):
     
    
        annotationlist=os.listdir(self.annfolder)
        name=annotationlist[idx]
        videolist=os.listdir(self.vid_dir)
        videoname=videolist[idx]
        video=osp.join(self.vid_dir,videoname)

        annotation=osp.join(self.annfolder,name)
        
        Kp=get_kp_timeseries(annotation)
        labels=get_labels(annotation)

        if (labels==1 ):
            labels=[0, 1]
        else:
            labels=[1, 0]
        
        
        
        bbox=get_bbox(annotation)
        
        
        

        if self.transform:
              _,_,Kp=chopper(Kp)
              Kp=normalize(Kp,annotation)
              Kp=kp_merger(Kp)
              Kp=zero_padding(Kp)
              Kp=torch.as_tensor(Kp, dtype=torch.float32)  
              labels=torch.as_tensor(labels, dtype=torch.float32) 

        return  Kp,labels

        
class LSTMClassifier(nn.Module):  
    #nn model class for clasification using lstm
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(LSTMClassifier,self).__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
        self.smax=nn.Softmax(dim=1)
    
    def forward(self,x):

        _,(hn,_)=self.lstm(x)
        out=self.fc(hn[-1,:,:])
        out=self.smax(out)
        return out
    
def train_model(self,train_loader,validation_loader,criterion,optimizer,num_epochs):
        #training function for lstm model
        epochs=[]
        log_dir = "logs/"  # Set the log directory
        correct_predictions = 0
        global_step = 0
        os.makedirs(log_dir, exist_ok=True)
        try:
            summary_writer = tf.summary.create_file_writer(log_dir)
        except: print('WARNING: could not create tensorboard summary writer')

        val_loss = 0.0
        l1_lambda = 0
        
       
        for epoch in range(num_epochs):
            epochs.append(epoch)
            correct_predictions = 0
            
            for data,labels in train_loader:

                l1_reg = torch.tensor(0.0)  # Initialize with zero
                for param in self.parameters():
                    l1_reg += torch.sum(torch.abs(param))
                # labels.squeeze_(0)
                self.train()
                
                data = data.permute(0,2,1)
                optimizer.zero_grad()
                outputs=self.forward(data)
                print (outputs,labels)
                
                loss=criterion(outputs,labels)+l1_lambda*l1_reg
                loss.backward()
                optimizer.step()
                # l1_reg.detach_()
                correct_predictions += (outputs-labels<0.005).sum().item()
                acc=100.0*correct_predictions/len(train_loader)
                print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.4f},accuracy={acc:.4f}')
                
                with summary_writer.as_default():
                    tf.summary.scalar("Train Loss", loss.item(), step=global_step)
                    tf.summary.scalar("Train Accuracy", acc, step=global_step)
                    tf.summary.scalar("Validation Loss", val_loss, step=global_step)

                global_step += 1


            self.eval()
            
            val_predictions = []
            val_targets = []
            for data,labels in validation_loader:
               
               data = data.permute(0,2,1)
               outputs=self.forward(data)
               val_loss = criterion(outputs, labels).item()
            
            print(f'epoch {epoch+1}/{num_epochs},val_loss={val_loss:.4f}')

        print('finished training')
        torch.save(self.state_dict(), 'saved_models/linear_model.pth')
    
def test_model(self,test_loader):
        with torch.no_grad():
            n_correct=0
            n_samples=0
            for data,labels in test_loader:
                data = data.permute(0,2,1)
                outputs=self.forward(data)
                print (outputs,labels)
                _,predictions=torch.max(outputs.data,1)
                n_samples+=labels.size(0)
                n_correct+=(predictions>0.7*labels).sum().item()
            acc=100.0*n_correct/n_samples
            print(f'accuracy={acc}')

def test(path,model,fullpose=True):
    model.eval()
    with torch.no_grad():

        if fullpose==True:
            Kp=get_kp_timeseries(path)
            _,_,Kp=chopper(Kp)
            Kp=normalize(Kp,path)
            Kp=kp_merger(Kp)
            Kp=zero_padding(Kp)
            Kp=torch.as_tensor(Kp, dtype=torch.float32)  
            Kp=torch.unsqueeze(Kp,0)
            outputs=model(Kp)
        else:
            Kp=get_kp_timeseries(path)
            _,_,Kp=chopper(Kp)
            Kp=normalize(Kp,path)
            feature=FTdescriptor(Kp)
            feature=zero_pad_features(feature)
            feature=torch.as_tensor(feature, dtype=torch.float32)
            feature=torch.unsqueeze(feature,0)
            feature=feature.permute(0,2,1)
            outputs=model(feature)
       

        label=get_labels(path)
        
       
        outputs=outputs.tolist()
       
        outputs=outputs[0]
        
        
        if len(outputs)==10:
            label=label*100
            m=max(outputs)
            prob=outputs.index(m)
            if prob==0:
                est='0-10%'
            elif prob==1:
                est='10-20%'
            elif prob==2:
                est='20-30%'
            elif prob==3:
                est='30-40%'
            elif prob==4:
                est='40-50%'
            elif prob==5:
                est='50-60%'
            elif prob==6:
                est='60-70%'
            elif prob==7:
                est='70-80%'
            elif prob==8:
                est='80-90%'
            elif prob==9:
                est='90-100%'
            pred_string='predicted shooting :' +est
            label_string='actual free throw % :' +str(label)
        elif len(outputs)==5:
            label=label*100
            m=max(outputs)
            prob=outputs.index(m)
            if prob==0:
                est='0-20%'
            elif prob==1:
                est='20-40%'
            elif prob==2:
                est='40-60%'
            elif prob==3:
                est='60-80%'
            elif prob==4:
                est='80-100%'
            pred_string='predicted shooting :' +est
            label_string='actual free throw % :' +str(label)
        elif len(outputs)==2:
            prob=outputs[0]
            
            if (outputs[1]>outputs[0]):
                string="IN with:  "
                prob=outputs[1]
                prob=float(prob)
                prob=round(prob,4)
                prob=prob*100
                prob=str(prob)
                print ('classified as a make with probability',prob)
                p_string='make'
            else:
                string="OUT with:  "
                prob=outputs[0]
                prob=float(prob)
                prob=round(prob,4)
                prob=prob*100
                prob=str(prob)
                print ('classified as a miss with probability',prob)
                p_string='miss'
            pred_string=string + prob+'%'+ ' confidence'
            
            if label==1:
                label_string=' actual result : make'
                l_string='make'
            else:   
                label_string=' actual result : miss'
                l_string='miss'

        
        

    return pred_string,label_string
      

        
def test_regression(path,video_path,model):
    model.eval()
    with torch.no_grad():
        Kp=get_kp_timeseries(path)
        _,_,Kp=chopper(Kp)
        Kp=normalize(Kp,path)
        # Kp=kp_merger(Kp)
        # Kp=zero_padding(Kp)
        feature=FTdescriptor(Kp)
        feature=zero_pad_features(feature)
        feature=torch.as_tensor(feature, dtype=torch.float32)
        Kp=torch.as_tensor(Kp, dtype=torch.float32)  
        Kp=torch.unsqueeze(Kp,0)
        
        feature=torch.unsqueeze(feature,0)
        feature=feature.permute(0,2,1)
     
        # Kp=Kp.permute(0,2,1)

        # outputs=model(Kp)
        outputs=model(feature)
        label=get_labels(path)
        label=label*100
       
        outputs=outputs.tolist()
    
        prob=outputs[0][0]
        print(outputs[0])
        prob=float(prob)
        prob=round(prob,4)
        prob=prob*100
        prob=str(prob)
            # prob=outputs.tolist()
        pred_string="predicted shooting :" + prob+'%'
        label_string='actual free throw % :' +str(label)


    return pred_string,label_string
        # frame_width = 640
       


class FTlineardataset(torch.utils.data.Dataset):
     
    #datset for linear regression and clasification using full pose array

    def __init__(self, annotations_folder,vid_dir, transform=True, target_transform=None):
        
        dataDir=vid_dir
        self.annfolder=annotations_folder
        self.vid_dir=vid_dir
        self.transform = transform
        
        



    def __len__(self):
        list=os.listdir(self.annfolder)
        
        return len(list)

    def __getitem__(self, idx):
     
    
        annotationlist=os.listdir(self.annfolder)
        name=annotationlist[idx]
        

        annotation=osp.join(self.annfolder,name)
        
        Kp=get_kp_timeseries(annotation)
        label=get_labels(annotation)

        
#clasification flag: 1 for 10 classes, 2 for 5 classes, 3 for binary else for regression.
#expecting labels either 1 or 0 or a float between 0 and 1

        classification=1
        if classification==1:
            labels=np.zeros(10)
            if label<0.1:
                labels[0]=1
                l=0
            elif label>=0.1 and label<0.2:
                labels[1]=1
                l=1
            elif label>=0.2 and label<0.3:
                labels[2]=1
                l=2
            elif label>=0.3 and label<0.4:
                labels[3]=1
                l=3
            elif label>=0.4 and label<0.5:
                labels[4]=1
                l=4
            elif label>=0.5 and label<0.6:
                labels[5]=1
                l=5
            elif label>=0.6 and label<0.7:
                labels[6]=1
                l=6
            elif label>=0.7 and label<0.8:
                labels[7]=1
                l=7
            elif label>=0.8 and label<0.9:
                labels[8]=1
                l=8
            elif label>=0.9:
                labels[9]=1
                l=9
        elif classification==2: 
            
            labels=[0,0,0,0,0]

            if label<0.2:
                labels[0]=1
                l=0
            elif label>=0.2 and label<0.4:
                labels[1]=1
                l=1
            elif label>=0.4 and label<0.6:
                labels[2]=1
                l=2
            elif label>=0.6 and label<0.8:
                labels[3]=1
                l=3
            elif label>=0.8:
                labels[4]=1
                l=4

        else:   
            labels=label
        bbox=get_bbox(annotation)
        
        
        

        if self.transform:
              _,_,Kp=chopper(Kp)
              Kp=normalize(Kp,annotation)
              Kp=kp_merger(Kp)
              Kp=zero_padding(Kp)
              Kp=torch.as_tensor(Kp, dtype=torch.float32)  
              labels=torch.as_tensor(labels, dtype=torch.float32) 

        return  Kp,labels
    
class LSTMlinearClassifier(nn.Module):  

    #nn model class for linear clasification/regression using lstm

    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(LSTMlinearClassifier,self).__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
        self.smax=nn.Softmax(dim=1)
        
    
    def forward(self,x):
    
        # Forward propagate LSTM
        _,(hn,_)=self.lstm(x)
        out=self.fc(hn[-1,:,:])
        # out=self.smax(out)
        return out

class FTdatasetparams(torch.utils.data.Dataset):
     
    #dataset class using engineered features

    def __init__(self, annotations_folder,vid_dir, transform=True, target_transform=None):
        
        dataDir=vid_dir
        self.annfolder=annotations_folder
        self.vid_dir=vid_dir
        self.transform = transform
        



    def __len__(self):
        list=os.listdir(self.annfolder)
        return len(list)

    def __getitem__(self, idx):
     
    
        annotationlist=os.listdir(self.annfolder)
        name=annotationlist[idx]
        annotation=osp.join(self.annfolder,name)

        Kp=get_kp_timeseries(annotation)
        label=get_labels(annotation)
        #clasification flag: 1 for 10 classes, 2 for 5 classes, 3 for binary else for regression.
        #expecting labels either 1 or 0 or a float between 0 and 1

        classification=5
        if classification==1:
            labels=np.zeros(10)
            if label<0.1:
                labels[0]=1
                l=0
            elif label>=0.1 and label<0.2:
                labels[1]=1
                l=1
            elif label>=0.2 and label<0.3:
                labels[2]=1
                l=2
            elif label>=0.3 and label<0.4:
                labels[3]=1
                l=3
            elif label>=0.4 and label<0.5:
                labels[4]=1
                l=4
            elif label>=0.5 and label<0.6:
                labels[5]=1
                l=5
            elif label>=0.6 and label<0.7:
                labels[6]=1
                l=6
            elif label>=0.7 and label<0.8:
                labels[7]=1
                l=7
            elif label>=0.8 and label<0.9:
                labels[8]=1
                l=8
            elif label>=0.9:
                labels[9]=1
                l=9
        elif classification==2: 
            
            labels=[0,0,0,0,0]

            if label<0.2:
                labels[0]=1
                l=0
            elif label>=0.2 and label<0.4:
                labels[1]=1
                l=1
            elif label>=0.4 and label<0.6:
                labels[2]=1
                l=2
            elif label>=0.6 and label<0.8:
                labels[3]=1
                l=3
            elif label>=0.8:
                labels[4]=1
                l=4

        elif classification==3: 
            
            if (label==1 ):
                labels=[0, 1]
            else:
                labels=[1, 0]

        else:
            labels=label


        if self.transform:
            _,_,Kp=chopper(Kp)
            Kp=normalize(Kp,annotation)
            Features=FTdescriptor(Kp)
            Features=zero_pad_features(Features)
            Features=torch.as_tensor(Features, dtype=torch.float32)  
            labels=torch.as_tensor(labels, dtype=torch.float32) 
           
        return  Features,labels