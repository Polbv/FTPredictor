import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp
import math

def get_kp_timeseries(path):
#returns the list of keypoints in the video
    with open(path,'r') as f:
        data = f.read()
    

    data = json.loads(data)
    head=[]#0
    center=[]#1
    dos=[]#2
    tres=[]#3
    cuatro=[]#4
    rshoulder=[]#6
    lshoulder=[]#5
    lelbow=[]#7
    relbow=[]#8
    lhand=[]#9
    rhand=[]#10
    lhip=[]#11
    rhip=[]#12
    lknee=[]#13
    rknee=[]#14
    lfoot=[]#15
    rfoot=[]#16
    n_frames=0
   
    for i in data:
        n_frames=n_frames+1
        if i==data[-1]:
            break
        instance=i['instances']
        instance=instance[0]
        KP=instance['keypoints']
        for j in range (0,len(KP)):
            if j==0:
                head.append(KP[j])
            elif j==1:
                center.append(KP[j])
            elif j==2:
                dos.append(KP[j])
            elif j==3:
                tres.append(KP[j])
            elif j==4:
                cuatro.append(KP[j])
            elif j==6:
                rshoulder.append(KP[j])
            elif j==5:
                lshoulder.append(KP[j])
            elif j==8:
                relbow.append(KP[j])
            elif j==7:
                lelbow.append(KP[j])
            elif j==10:
                rhand.append(KP[j])
            elif j==9:
                lhand.append(KP[j])
            elif j==16:
                rfoot.append(KP[j])
            elif j==15:
                lfoot.append(KP[j])
            elif j==14:
                rknee.append(KP[j])
            elif j==13:
                lknee.append(KP[j])
            elif j==12:
                rhip.append(KP[j])
            elif j==11:
                lhip.append(KP[j])
            else: pass
    
    KP_list=[head,center,dos,tres,cuatro,lshoulder,rshoulder,lelbow,relbow,lhand,rhand,lhip,rhip,lknee,rknee,lfoot,rfoot]
    KP_list=yinverter(KP_list)
    return KP_list

def get_labels(path):
#reads labels form annotation path

    with open(path,'r') as json_file:
        data = json.load(json_file)
        label=data[-1]
        label=label['label']
        
    return label

def get_bbox(path):

    with open(path,'r') as f:
        data = f.read()
    data = json.loads(data)
    n_frames=0
    bbox=[]
    for i in data:
        if i==data[-1]:
            break
        instance=i['instances']
        instance=instance[0]
        bbox.append(instance['bbox'])
        n_frames=n_frames+1

    return bbox

def normalize(Kpseq,path):
    #normalizes the keypoints
    #assuming bbox in x1y1x2y2 format
    bboxes=get_bbox(path)
    bboxx=[]
    bboxy=[]
    for box in bboxes:
        width=box[0][2]-box[0][0]
        heigth=box[0][3]-box[0][1]
        
        bboxx.append(box[0][1])
        bboxy.append((1080-box[0][0])-(1080-box[0][2]))
    
    cornerx=min(bboxx)
    cornery=min(bboxy)
    corner=[cornerx,cornery]
    corner3=[corner[0]-heigth,corner[1]]
    
    Kpsequence=[]
    for kp in Kpseq:
        for i in range(0,len(kp)):
            kp[i][0]=kp[i][0]-corner[0]
            kp[i][1]=kp[i][1]-corner[1]
        Kpsequence.append(kp)
    

    return Kpsequence

def yinverter(KP):
    #inverts the y axis
    headseq=KP[0]
    length=len(headseq)
    
    for i in range(0,len(KP)):
        
        for j in range(0,length):
            
            KP[i][j][1]=1080-KP[i][j][1]
    # print (KP)

    return KP

def chopper(Kpseq):
     
    # bounds the keypoints to the start frame of FT motion and last frame.
    # starting frame defined as lowest point of rhip and last frame defined as highest point of rhand(release)
    id=keypoint_id('rhand')
    rhandseq=Kpseq[id]
    rhandx=[]
    rhandy=[]
    
    
    for x,y in rhandseq:
        rhandx.append(x)
        rhandy.append(y)
    n_frames=len(rhandx)
    minx=min(rhandx)
    maxx=max(rhandx)
    miny=min(rhandy)
    maxy=max(rhandy)
    


    ending_frame=rhandy.index(maxy)

    id=keypoint_id('rhip')
    rhipseq=Kpseq[4]
    rhipx=[]
    rhipy=[]
    
    for x,y in rhipseq:
            rhipx.append(x)
            rhipy.append(y)

    minx=min(rhipx)
    maxx=max(rhipx)
    miny=min(rhipy)
    maxy=max(rhipy)


    starting_frame=rhipy.index(miny)

    
    
    
    for i in range(0,len(Kpseq)):
        Kpseq[i]=Kpseq[i][starting_frame:ending_frame]

        
   

    
    return starting_frame,ending_frame,Kpseq

def plotter(Kpseq,name):
    #a function to plot the  the position of a given keypoint over time
    #Kpseq is a list of lists
    #name is the name of the keypoint
    #Kpseq[0] is the first keypoint
    #Kpseq[0][0] is the first coordinates of the first keypoint
    #Kpseq[0][0][0] is the x coordinate of the first coordinate of the first keypoint
    #Kpseq[0][0][1] is the y coordinate of the first coordinate of the first keypoint
    #Kpseq[0][1][0] is the x coordinate of the second coordinate of the first keypoint
    #Kpseq[0][1][1] is the y coordinate of the second coordinate of the first keypoint
   
   
    id=keypoint_id(name)
    print(id)
    seq=Kpseq[id]
    xseq=[]
    yseq=[]
    t=[]
    i=0
    for y,x in seq:
        xseq.append(x)
        yseq.append(y)
        t.append(i)
        i=i+1
    plt.plot(t,xseq)
    plt.xlabel("frames")
    plt.ylabel("lateral movement")
    plt.title(name)
    plt.show()
    plt.plot(t,yseq)    
    plt.xlabel("frames")
    plt.ylabel("vertical movement")
    plt.title(name)
    plt.show()
    plt.plot(xseq,yseq) 
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(name)
    plt.show()

def keypoint_id(name):
        
            if name=='head':
                id=0
            elif name=='center':
                id=1
            elif name=='rshoulder':
                id=6
            elif name=='lshoulder':
                id=5
            elif name=='relbow':
                id=8
            elif name=='lelbow':
                id=7
            elif name=='rhand':
                id=10
            elif name=='lhand':
                id=9
            elif name=='rfoot':
                id=16
            elif name=='lfoot':
                id=15
            elif name=='rknee':
                id=14
            elif name=='lknee':
                id=13
            elif name=='rhip':
                id=12
            elif name=='lhip':
                id=11
            else:
                id=-1
            assert id!=-1, "keypoint name not found"    
            return id

def plotter_all(Kpseq):
    #plots rhand,rhip,head,rshoulder,rknee over time on the same graph
    rhandseq=Kpseq[10]
    rhandx=[]
    rhandy=[]
    t=[]
    i=0
    for y,x in rhandseq:
        rhandx.append(x)
        rhandy.append(y)
        t.append(i)
        i=i+1
    rhipseq=Kpseq[12]
    rhipx=[]
    rhipy=[]

    for y,x in rhipseq:
        rhipx.append(x)
        rhipy.append(y)

    headseq=Kpseq[0]
    headx=[]
    heady=[]
    for y,x in headseq:
        headx.append(x)
        heady.append(y)

    rshoulderseq=Kpseq[6]
    rshoulderx=[]
    rshouldery=[]
    for y,x in rshoulderseq:
        rshoulderx.append(x)
        rshouldery.append(y)

    rkneesq=Kpseq[14]
    rkneex=[]
    rkneey=[]
    for y,x in rkneesq:
        rkneex.append(x)
        rkneey.append(y)
    
    # plotting the points
    plt.plot(rhandy, rhandx, color='green', linestyle='dashed', linewidth = 3,label='rhand')
    plt.plot(rhipy, rhipx, color='red', linestyle='dashed', linewidth = 3,label='rhip')
    plt.plot(heady, headx, color='blue', linestyle='dashed', linewidth = 3,label='head')
    plt.plot(rshouldery, rshoulderx, color='black', linestyle='dashed', linewidth = 3,label='rshoulder')
    plt.plot(rkneey, rkneex, color='yellow', linestyle='dashed', linewidth = 3,label='rknee')
    plt.axis([400, 1000,100, 700])
    plt.xlabel("frames")
    plt.ylabel("y")
    plt.title("y movement") 
    plt.legend()
    plt.show()

def initial_pose(Kp,path,time=0):
    #plots the initial pose at given time
    # path is necesary to get the bbox if uncommented


    rhandseq=Kp[10]
    rhandx=rhandseq[time][0]
    rhandy=rhandseq[time][1]
    lhandseq=Kp[9]
    lhandx=lhandseq[time][0]
    lhandy=lhandseq[time][1]

    rhipseq=Kp[12]
    rhipx=rhipseq[time][0]
    rhipy=rhipseq[time][1]
    lhipseq=Kp[11]
    lhipx=lhipseq[time][0]
    lhipy=lhipseq[time][1]

    headseq=Kp[0]
    headx=headseq[time][0]
    heady=headseq[time][1]
    
    rshoulderseq=Kp[6]
    rshoulderx=rshoulderseq[time][0]
    rshouldery=rshoulderseq[time][1]
    lshoulderseq=Kp[5]
    lshoulderx=lshoulderseq[time][0]
    lshouldery=lshoulderseq[time][1]

    relbowseq=Kp[8]
    relbowx=relbowseq[time][0]
    relbowy=relbowseq[time][1]
    lelbowseq=Kp[7]
    lelbowx=lelbowseq[time][0]
    lelbowy=lelbowseq[time][1]



    rkneesq=Kp[14]
    rkneex=rkneesq[time][0]
    rkneey=rkneesq[time][1]
    lkneesq=Kp[13]
    lkneex=lkneesq[time][0]
    lkneey=lkneesq[time][1]

    rfootseq=Kp[16]
    rfootx=rfootseq[time][0]
    rfooty=rfootseq[time][1]
    lfootseq=Kp[15]
    lfootx=lfootseq[time][0]
    lfooty=lfootseq[time][1]


    x=[rhandx,lhandx,rhipx,lhipx,headx,rshoulderx,lshoulderx,relbowx,lelbowx,rkneex,lkneex,rfootx,lfootx]
    y=[rhandy,lhandy,rhipy,lhipy,heady,rshouldery,lshouldery,relbowy,lelbowy,rkneey,lkneey,rfooty,lfooty]
    l=len(x)
    labels=['rhand','lhand','rhip','lhip','head','rshoulder','lshoulder','relbow','lelbow','rknee','lknee','rfoot','lfoot']
    colors =[np.random.rand(3,) for i in range(l)]
    colors=[np.array([0.72864398, 0.84310565, 0.01784724]), np.array([0.48496909, 0.32457186, 0.58015776]), np.array([0.95817394, 0.03637732, 0.52790931]), np.array([0.48004414, 0.503227  , 0.09021211]), np.array([0.6540798 , 0.19121591, 0.19290773]), np.array([0.86534958, 0.2673518 , 0.66951117]), np.array([0.24003167, 0.38129286, 0.67093438]), np.array([0.99087251, 0.73938575, 0.26344408]), np.array([0.3818175 , 0.11925698, 0.77981263]), np.array([0.69230346, 0.47798787, 0.60643534]), np.array([0.80983081, 0.97694621, 0.25614411]), np.array([0.18105688, 0.8708405 , 0.08404625]), np.array([0.86549172, 0.71850368, 0.54745672])]
    
    fig, ax = plt.subplots()

    ##printing keypoints

    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=colors[i], label=labels[i])
    plt.xlabel("x")
    plt.axis([400, 1000,100, 700])

    ##printing box

    # box=get_bbox(path)
    # corner3=[box[0][time][0],box[0][time][1]]#fila,columna
    # # corner3=[corner3[0],1080-corner3[1]-(1080-box[0][time][2])]#xy
    # corner3=[corner3[0],1080-corner3[1]-(1080-box[0][time][2])]
    # height=box[0][time][2]-box[0][time][0]
    # width=box[0][time][3]-box[0][time][1]
    # box = plt.Rectangle(corner3,height,width, fill=False, color='blue')
    # plt.gca().add_patch(box)
    
    ##printing lines
   
    line1x=[rfootx,rkneex,rhipx,rshoulderx,relbowx,rhandx]
    line1y=[rfooty,rkneey,rhipy,rshouldery,relbowy,rhandy]
    line2x=[lfootx,lkneex,lhipx,lshoulderx,lelbowx,lhandx]
    line2y=[lfooty,lkneey,lhipy,lshouldery,lelbowy,lhandy]
    line3x=[rshoulderx,headx]
    line3y=[rshouldery,heady]
    line4x=[lshoulderx,headx]
    line4y=[lshouldery,heady]
    line5x=[rshoulderx,lshoulderx]
    line5y=[rshouldery,lshouldery]
    plt.plot(line1x,line1y,color='black')
    plt.plot(line2x,line2y,color='gray')
    plt.plot(line3x,line3y,color='gray')
    plt.plot(line4x,line4y,color='gray')
    plt.plot(line5x,line5y,color='black')





    plt.ylabel("y")
    plt.title("pose") 
    plt.legend()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Convert the image to BGR format (required by OpenCV)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



    plt.show()
    plt.close(fig)
    return img_bgr
    
def labellizer(path):
    #adds label to json file

        print (osp.dirname(path))
        parentpath=osp.dirname(path)
        print (osp.dirname(path))
        if osp.dirname(parentpath)=='data/predictions/in':

            data={
                "label":1
            }
        elif osp.dirname(parentpath)=='data/predictions/out':
            data={
                "label":0
            }
        else :
            raise Exception("error")
            print("error")
        
        with open(path,'r') as json_file:
            current_data=json.load(json_file)
            # current_data.update(data)
            new_data=current_data
            new_data.append(data)
            # print(new_data)
        with open(path,'w') as json_file:
            json.dump(new_data,json_file)

def zero_padding(Kp):
#zero pads sequences to the max length
    max_length=62 # change according to dataset
    for i in range(0,len(Kp)):
        length=len(Kp[i])
        while length<max_length:

            Kp[i].append(0)
            length=len(Kp[i])
    return Kp

def max_length(folder):
    #returns the max length of a sequence in a folder
    max_length=0
    for annotation in os.listdir('data/predictions/out/json'):
        path=osp.join('data/predictions/out/json',annotation)
        

        Kp=get_kp_timeseries(path)
        _,_,Kp=chopper(Kp)

        length=len(Kp[0])
        print(length)
        if length>max_length:
            max_length=length
    for annotation in os.listdir('data/predictions/in/json'):
        path=osp.join('data/predictions/on/json',annotation)
        

        Kp=get_kp_timeseries(path)
        _,_,Kp=chopper(Kp)

        length=len(Kp[0])
        print(length)
        if length>max_length:
            max_length=length
            
    return max_length

def kp_merger(Kp):

#converts 17x2 coordinate array to 34x1 array
     

    headseq=Kp[0]
    headx=[]
    heady=[]
    for coord in headseq:
        headx.append(coord[0])
        heady.append(coord[1])

    

    
    centerseq=Kp[1]
    centerx=[]
    centery=[]
    for coord in centerseq:
        centerx.append(coord[0])
        centery.append(coord[1])

   

    dosseq=Kp[2]
    dosx=[]
    dosy=[]
    for coord in dosseq:
        dosx.append(coord[0])
        dosy.append(coord[1])

    
    tresseq=Kp[3]
    tresx=[]
    tresy=[]
    for coord in tresseq:
        tresx.append(coord[0])
        tresy.append(coord[1])

    
    cuatroseq=Kp[4]
    cuatrox=[]
    cuatroy=[]
    for coord in cuatroseq:
        cuatrox.append(coord[0])
        cuatroy.append(coord[1])

    lshoulderseq=Kp[5]
    lshoulderx=[]
    lshouldery=[]
    for coord in lshoulderseq:
        lshoulderx.append(coord[0])
        lshouldery.append(coord[1])

    rshoulderseq=Kp[6]
    rshoulderx=[]
    rshouldery=[]
    for coord in rshoulderseq:
        rshoulderx.append(coord[0])
        rshouldery.append(coord[1])

    lelbowseq=Kp[7]
    lelbowx=[]
    lelbowy=[]
    for coord in lelbowseq:
        lelbowx.append(coord[0])
        lelbowy.append(coord[1])

    relbowseq=Kp[8]
    relbowx=[]
    relbowy=[]
    for coord in relbowseq:
        relbowx.append(coord[0])
        relbowy.append(coord[1])

    lhandseq=Kp[9]
    lhandx=[]
    lhandy=[]
    for coord in lhandseq:
        lhandx.append(coord[0])
        lhandy.append(coord[1])
    
    rhandseq=Kp[10]
    rhandx=[]
    rhandy=[]
    for coord in rhandseq:
        rhandx.append(coord[0])
        rhandy.append(coord[1])
    
    lhipseq=Kp[11]
    lhipx=[]
    lhipy=[]
    for coord in lhipseq:
        lhipx.append(coord[0])
        lhipy.append(coord[1])
    
    rhipseq=Kp[12]
    rhipx=[]
    rhipy=[]
    for coord in rhipseq:
        rhipx.append(coord[0])
        rhipy.append(coord[1])
    
    lkneeseq=Kp[13]
    lkneex=[]
    lkneey=[]
    for coord in lkneeseq:
        lkneex.append(coord[0])
        lkneey.append(coord[1])
    
    rkneeseq=Kp[14]
    rkneex=[]
    rkneey=[]
    for coord in rkneeseq:
        rkneex.append(coord[0])
        rkneey.append(coord[1])
    
    lfootseq=Kp[15]
    lfootx=[]
    lfooty=[]
    for coord in lfootseq:
        lfootx.append(coord[0])
        lfooty.append(coord[1])
    
    rfootseq=Kp[16]
    rfootx=[]
    rfooty=[]
    for coord in rfootseq:
        rfootx.append(coord[0])
        rfooty.append(coord[1])
    




    kp_new=[headx,heady,centerx,centery,dosx,dosy,tresx,tresy,cuatrox,cuatroy,lshoulderx,lshouldery,rshoulderx,rshouldery,lelbowx,lelbowy,relbowx,relbowy,lhandx,lhandy,rhandx,rhandy,lhipx,lhipy,rhipx,rhipy,lkneex,lkneey,rkneex,rkneey,lfootx,lfooty,rfootx,rfooty]
    
    return kp_new

def zero_pad_features(features):
    #zero pads sequences of features to the max length
    max_length=62
   
    
    for i in range(0,len(features)):
        length=len(features[i])
        while length<max_length:
            # Kp[i].append([0,0])
            features[i].append(0)
            length=len(features[i])
    return features

def FTdescriptor(Kpseq):
    #calculates features for FT motion

    hand_id=keypoint_id('rhand')
    hip_id=keypoint_id('rhip')
    shoulder_id=keypoint_id('rshoulder')
    knee_id=keypoint_id('rknee')
    foot_id=keypoint_id('rfoot')
    elbow_id=keypoint_id('relbow')
    head_id=keypoint_id('head')
    leg_ang=[]
    elb_ang=[]
    armpit_ang=[]
    head=[]
    dt=1/30

    for i in range(0,len(Kpseq[head_id])):
        headx=Kpseq[head_id][i][0]
        heady=Kpseq[head_id][i][1]
        head.append(heady)




    for i in range(0,len(Kpseq[hip_id])):
        
        kneex=Kpseq[knee_id][i][0]
        kneey=Kpseq[knee_id][i][1]
        footx=Kpseq[foot_id][i][0]
        footy=Kpseq[foot_id][i][1]
        hipx=Kpseq[hip_id][i][0]
        hipy=Kpseq[hip_id][i][1]
        tibia=(kneex-footx, kneey-footy)
        femur=(hipx-kneex, hipy-kneey)
        dot=tibia[0]*femur[0]+tibia[1]*femur[1]
        tibia_norm=math.sqrt(tibia[0]**2+tibia[1]**2)
        femur_norm=math.sqrt(femur[0]**2+femur[1]**2)
        ang=math.acos(dot/(tibia_norm*femur_norm))
        
        leg_ang.append(math.degrees(ang))




    for i in range(0,len(Kpseq[hip_id])):
            
            hipx=Kpseq[hip_id][i][0]
            hipy=Kpseq[hip_id][i][1]
            handx=Kpseq[hand_id][i][0]
            handy=Kpseq[hand_id][i][1]
            elbowx=Kpseq[elbow_id][i][0]
            elbowy=Kpseq[elbow_id][i][1]
            shoulderx=Kpseq[shoulder_id][i][0]
            shouldery=Kpseq[shoulder_id][i][1]

            radius=(handx-elbowx, handy-elbowy)
            humerus=(elbowx-shoulderx, elbowy-shouldery)
            torso=(shoulderx-hipx, shouldery-hipy)

            dot1=radius[0]*humerus[0]+radius[1]*humerus[1]
            dot2=radius[0]*torso[0]+radius[1]*torso[1]

            torso_norm=math.sqrt(torso[0]**2+torso[1]**2)
            radius_norm=math.sqrt(radius[0]**2+radius[1]**2)
            humerus_norm=math.sqrt(humerus[0]**2+humerus[1]**2)

            ang1=math.acos(dot1/(radius_norm*humerus_norm))
            ang2=math.acos(dot2/(radius_norm*torso_norm))

            elb_ang.append(math.degrees(ang1))
            armpit_ang.append(math.degrees(ang2))

    v_leg_ang=[0]
    v_elb_ang=[0]
    v_armpit_ang=[0]
    for i in range(0,len(Kpseq[head_id])-1):
        v_leg_ang.append((leg_ang[i+1]-leg_ang[i])/dt)
        v_elb_ang.append((elb_ang[i+1]-elb_ang[i])/dt)
        v_armpit_ang.append((armpit_ang[i+1]-armpit_ang[i])/dt)



    anglist=[head,leg_ang,elb_ang,armpit_ang,v_leg_ang,v_elb_ang,v_armpit_ang]

    return anglist

def feature_plotter(feature_vector):

    #plots the features over time
    heady=feature_vector[0]
    leg_ang=feature_vector[1]
    elb_ang=feature_vector[2]
    armpit_ang=feature_vector[3]
    v_leg_ang=feature_vector[4]
    v_elb_ang=feature_vector[5]
    v_armpit_ang=feature_vector[6]
    print(heady,len(heady))
    time=np.arange(0,len(heady))
    print(time)
    plt.plot(time,heady,linewidth=2.0, color='black',label='heady')
    plt.plot(time,leg_ang,linewidth=2.0, color='red',label='leg_ang')
    plt.plot(time,elb_ang,linewidth=2.0, color='blue',label='elb_ang')
    plt.plot(time,armpit_ang,linewidth=2.0, color='green',label='armpit_ang')
    # plt.plot(time,v_leg_ang,linewidth=2.0, color='orange',label='v_leg_ang')
    # plt.plot(time,v_elb_ang,linewidth=2.0, color='purple',label='v_elb_ang')
    # plt.plot(time,v_armpit_ang,linewidth=2.0, color='yellow',label='v_armpit_ang')
    plt.legend()
    plt.show()

def feature_comparison(feature_vector,feature_vector2,feature_vector3):
    #comparison between 3 feature vectors
    heady=feature_vector[0]
    leg_ang=feature_vector[1]
    elb_ang=feature_vector[2]
    armpit_ang=feature_vector[3]
    v_leg_ang=feature_vector[4]
    v_elb_ang=feature_vector[5]
    v_armpit_ang=feature_vector[6]
    time=np.arange(0,len(heady))
    # plt.plot(time,heady,linewidth=2.0, color='black',label='heady')
    plt.plot(time,leg_ang,linewidth=2.0, color='red',label='leg_ang')
    plt.plot(time,elb_ang,linewidth=2.0, color='blue',label='elb_ang')
    plt.plot(time,armpit_ang,linewidth=2.0, color='green',label='armpit_ang')
    
    heady=feature_vector2[0]
    leg_ang=feature_vector2[1]
    elb_ang=feature_vector2[2]
    armpit_ang=feature_vector2[3]
    v_leg_ang=feature_vector2[4]
    v_elb_ang=feature_vector2[5]
    v_armpit_ang=feature_vector2[6]
    time=np.arange(0,len(heady))
    # plt.plot(time,heady,linewidth=2.0,linestyle='--', color='black',label='heady')
    plt.plot(time,leg_ang,linewidth=2.0,linestyle='--', color='red',label='leg_ang')
    plt.plot(time,elb_ang,linewidth=2.0,linestyle='--', color='blue',label='elb_ang')
    plt.plot(time,armpit_ang,linewidth=2.0,linestyle='--', color='green',label='armpit_ang')
    plt.title('S1_2 vs S2_3')
    plt.xlabel('Frames')
    plt.ylabel('Angles in degrees')
    plt.legend()
    plt.show()

    heady=feature_vector[0]
    leg_ang=feature_vector[1]
    elb_ang=feature_vector[2]
    armpit_ang=feature_vector[3]
    v_leg_ang=feature_vector[4]
    v_elb_ang=feature_vector[5]
    v_armpit_ang=feature_vector[6]
    time=np.arange(0,len(heady))
    # plt.plot(time,heady,linewidth=2.0, color='black',label='heady')
    plt.plot(time,leg_ang,linewidth=2.0, color='red',label='leg_ang')
    plt.plot(time,elb_ang,linewidth=2.0, color='blue',label='elb_ang')
    plt.plot(time,armpit_ang,linewidth=2.0, color='green',label='armpit_ang')
    time=np.arange(0,len(heady))
    # plt.plot(time,heady,linewidth=2.0, color='black',label='heady')
    plt.plot(time,leg_ang,linewidth=2.0, color='red',label='leg_ang')
    plt.plot(time,elb_ang,linewidth=2.0, color='blue',label='elb_ang')
    plt.plot(time,armpit_ang,linewidth=2.0, color='green',label='armpit_ang')
    heady=feature_vector3[0]
    leg_ang=feature_vector3[1]
    elb_ang=feature_vector3[2]
    armpit_ang=feature_vector3[3]
    v_leg_ang=feature_vector3[4]
    v_elb_ang=feature_vector3[5]
    v_armpit_ang=feature_vector2[6]
    time=np.arange(0,len(heady))
    # plt.plot(time,heady,linewidth=2.0,linestyle='--', color='black',label='heady')
    plt.plot(time,leg_ang,linewidth=2.0,linestyle='--', color='red',label='leg_ang')
    plt.plot(time,elb_ang,linewidth=2.0,linestyle='--', color='blue',label='elb_ang')
    plt.plot(time,armpit_ang,linewidth=2.0,linestyle='--', color='green',label='armpit_ang')
    plt.title('S1_2 vs S1_3')
    plt.xlabel('Frames')
    plt.ylabel('Angles in degrees')
    plt.legend()
    plt.show()

   
    averages1=stats(feature_vector)
    averages2=stats(feature_vector2)
    averages3=stats(feature_vector3)
    ind=1
    bar_width = 0.25
    categories = ['Media', 'desviación', 'mínimo', 'máximo']
    x=np.arange(len(categories))
    values1 = [averages1[0][ind] , averages1[1][ind], averages1[2][ind], averages1[3][ind]]
    values2 = [averages2[0][ind] , averages2[1][ind], averages2[2][ind], averages2[3][ind]]
    values3 = [averages3[0][ind] , averages3[1][ind], averages3[2][ind], averages3[3][ind]]
    plt.bar(x - bar_width, values1, bar_width, label='S1_2', color='r')
    plt.bar(x, values2, bar_width, label='S2_3', color='g')
    plt.bar(x + bar_width, values3, bar_width, label='S1_3', color='b')
    plt.xlabel('Averages')
    plt.ylabel('Values')
    plt.title('promedio de ángulo de rodilla')
    plt.xticks(x, categories)
    plt.legend()
    plt.show()
    ind=2
    bar_width = 0.25
    categories = ['Media', 'desviación', 'mínimo', 'máximo']
    x=np.arange(len(categories))
    values1 = [averages1[0][ind] , averages1[1][ind], averages1[2][ind], averages1[3][ind]]
    values2 = [averages2[0][ind] , averages2[1][ind], averages2[2][ind], averages2[3][ind]]
    values3 = [averages3[0][ind] , averages3[1][ind], averages3[2][ind], averages3[3][ind]]
    plt.bar(x - bar_width, values1, bar_width, label='S1_2', color='r')
    plt.bar(x, values2, bar_width, label='S2_3', color='g')
    plt.bar(x + bar_width, values3, bar_width, label='S1_3', color='b')
    plt.xlabel('Averages')
    plt.ylabel('Values')
    plt.title('promedio de ángulo de codo')
    plt.xticks(x, categories)
    plt.legend()
    plt.show()
    ind=3
    bar_width = 0.25
    categories = ['Media', 'desviación', 'mínimo', 'máximo']
    x=np.arange(len(categories))
    values1 = [averages1[0][ind] , averages1[1][ind], averages1[2][ind], averages1[3][ind]]
    values2 = [averages2[0][ind] , averages2[1][ind], averages2[2][ind], averages2[3][ind]]
    values3 = [averages3[0][ind] , averages3[1][ind], averages3[2][ind], averages3[3][ind]]
    plt.bar(x - bar_width, values1, bar_width, label='S1_2', color='r')
    plt.bar(x, values2, bar_width, label='S2_3', color='g')
    plt.bar(x + bar_width, values3, bar_width, label='S1_3', color='b')
    plt.xlabel('Averages')
    plt.ylabel('Values')
    plt.title('promedio de ángulo de axila')
    plt.xticks(x, categories)
    plt.legend()
    plt.show()
    
def stats(features):
    # features=features[:4] #remove the first row of zeros
    # Calculate mean, standard deviation, minimum, and maximum for each feature
    mean = [np.mean(feature) for feature in features]
    std = [np.std(feature) for feature in features]
    min_val = [np.min(feature) for feature in features]
    max_val = [np.max(feature) for feature in features]
    n=[len(feature) for feature in features]
   
    # Stack the statistics for each feature into a single list
    statistics = [mean, std, min_val, max_val]
    # statistics=torch.as_tensor(statistics, dtype=torch.float32)  
    return statistics