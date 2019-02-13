#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 02:36:52 2019

@author: vishay
"""

# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
#import imutils
import cv2
from scipy import ndimage
import scipy.io
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # The GPU id to use, usually either "0" or "1"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" ## here you can give 0 or 1 based on whatever gpu is unused



# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
#	help="path to trained model model")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

subject_names=['jyotirmoy_das','nikhil_bhardwaj','manoj','renuka']
distance = ['1ft','1.5ft']

print("[INFO] loading network...")
model = load_model('/home/vishay/Music/CNN/cnn.h5')

q =  list(range(0,16))
p = list(range(1))



for subject_name in subject_names:
    angle = []
    topleft = []
    bottomright = []
    HRloc = open('/media/vishay/ExternalHDD/DATA/Complete/'+subject_name+'/HRloc.txt', 'r').read().splitlines()
    for i in HRloc:
        topleft.append(list(map(int,i.split(' ')[0:2])))
        bottomright.append(list(map(int,i.split(' ')[2:4])))
        angle.append(int(i.split(' ')[4]))
    tl = np.reshape(topleft,(16,2))
    br = np.reshape(bottomright,(16,2))
    a = np.reshape(angle,(16))
    for j in p:
    
        for i in q:
            index = str(i+1)
           #print(index)
            jay = str(j)
        
            if i<9:	
                	vidcap = cv2.VideoCapture('/media/vishay/ExternalHDD/DATA/Complete/'+subject_name+'/samsung/'+distance[j]+'/'+'0%s.mp4' %index)
            else:
            	    vidcap = cv2.VideoCapture('/media/vishay/ExternalHDD/DATA/Complete/'+subject_name+'/samsung/'+distance[j]+'/'+'%s.mp4' %index)
            count = 1    
            success = True      
    #        angle = 90
            left = []
            right = []
        
            while success:
                success,image = vidcap.read()
                
                if success == False:
                    break
                                               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = ndimage.rotate(gray,a[i])
                hr = gray[tl[i,1]:br[i,1],tl[i,0]:br[i,0]]    
                width, height = hr.shape[::-1]
                b1 = hr[:,0:round(width/2) + 1];
                b2 = hr[:,round(width/2) - 1  : width+1];
                b1 = cv2.resize(b1, (28, 28))
                _,b1 = cv2.threshold(b1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                b1 = b1.astype("float") / 255.0
                b1 = img_to_array(b1)
                b1 = np.expand_dims(b1, axis=0)
                b2 = cv2.resize(b2, (28, 28))
                _,b2 = cv2.threshold(b2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                b2 = b2.astype("float") / 255.0
                b2 = img_to_array(b2)
                b2 = np.expand_dims(b2, axis=0)
                count += 1
                left.append(np.argmax(model.predict(b1)[0]))
                right.append(np.argmax(model.predict(b2)[0]))
                if count%100==0:
                    print('frame #'+str(count)+' of video #'+index+' distance '+jay+' '+subject_name)
                
            left = np.array(left)
            right = np.array(right)
            num = 10*left+right
            if i<9:	
                	scipy.io.savemat('/media/vishay/ExternalHDD/DATA/Complete/'+subject_name+'/samsung/'+distance[j]+'/'+'GT/'+'0'+index+'.mat', {'gt':num})
            else:
            	    scipy.io.savemat('/media/vishay/ExternalHDD/DATA/Complete/'+subject_name+'/samsung/'+distance[j]+'/'+'GT/'+index+'.mat', {'gt':num})