#author:Zhitian Zhang
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#deactivate tensorflow warning
#import numpy as np
#import pandas as pd
import tensorflow as tf
from pytube import YouTube
import cv2
#import sys
import subprocess
from html.parser import HTMLParser
import urllib.request

f = open("./index.htm", "r")
lines=f.readlines()

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

nimage=[]

n1 = 1#frist tfrecord file to download
n2 = 5 #last tfrecord file to download

for i in range(n1,n2):
    tfrecord = strip_tags(lines[i])
    tfrecord = tfrecord.replace("\n","")
    urllib.request.urlretrieve("http://us.data.yt8m.org/1/video_level/train/%s" %tfrecord,"%s" %tfrecord)
    tffile =("./%s" %tfrecord)
    print ('------------------------------------')
    print ('File:%s downloaded' %tfrecord)
    #label = pd.read_csv('./label_names.csv')
    car = 4
    car_vid = []
    #more labels for video here
    labels = []
    totalf = 0 
    n = 0
    #start reading data from tf record file
    for example in tf.python_io.tf_record_iterator(tffile):
        tf_example = tf.train.Example.FromString(example)
        label_temp = tf_example.features.feature['labels'].int64_list.value
        labels.append(label_temp)
        print ('Processing videos in tfrecord file number %d' %i)
        if car in label_temp:
            vid_temp = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
            car_vid.append(vid_temp)
            try:
                yt = YouTube("http://www.youtube.com/watch?v=%s" %vid_temp)
                yt.set_filename('tempvideo')
                video = yt.get('mp4','360p')
                video.download('./temp')
            except:
                continue
            else:     
                vd = cv2.VideoCapture('./temp/tempvideo.mp4')
                success,image = vd.read()
                count = 0
                nframe = 0
                success = True
                time = 2000
                #print ('processing video number %d'%n)
                n += 1
                while count<20:
                    vd.set(cv2.CAP_PROP_POS_MSEC,time)
                    success,image = vd.read()
                    if success:
                        resized_image = cv2.resize(image,(320,180))
                    else:
                        count = 20
                        continue
                    cv2.imwrite("./output/car/frame%d%s.jpg" %(count,vid_temp), resized_image)
                    imagename = ("./output/car/frame%d%s.jpg" %(count,vid_temp))
                    pred = subprocess.run(['python','classify_image.py','--image_file', '%s' %imagename], stdout=subprocess.PIPE).stdout.decode('utf-8')
                    #os.system("python classify_image.py --image_file %s" %imagename)
                    pred = pred.replace("[","")
                    pred = pred.replace("]","")
                    pred = pred.replace("'","")
                    pred = pred.replace(",","")
                    pred = pred.split()
                    #print (pred)
                    if "car" in pred:
                        time += 5000
                        count +=1
                        nframe +=1
                    else:
                        time += 5000
                        count +=1
                        os.remove(imagename)
                os.remove('./temp/tempvideo.mp4')
                totalf = totalf + nframe
                #print ('number of frames saved for this video: %d' %nframe)
    print ('Number of images for this file:%d' %totalf)
    print ('Finishing and deleting file: %s' %tfrecord)
    print ('---------------------------------------------------')
    nimage.append(totalf)
    os.remove(tffile)
        
    
        
    

