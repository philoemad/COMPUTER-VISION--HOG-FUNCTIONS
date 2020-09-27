import os,codecs,numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle 
import cv2
# code for load data set
datapath = 'C:/Users/philo/Downloads/v/' #path of the extrated file

files = os.listdir(datapath)

def get_int(b):   
    return int(codecs.encode(b, 'hex'), 16)

data_dict = {}
for file in files:
    if file.endswith('ubyte'): 
        print('Reading ',file)
        with open (datapath+file,'rb') as f:
            data = f.read()
            type = get_int(data[:4])   
            length = get_int(data[4:8]) 
            if (type == 2051):
                category = 'images'
                num_rows = get_int(data[8:12]) 
                num_cols = get_int(data[12:16])  
                parsed = numpy.frombuffer(data,dtype = numpy.uint8, offset = 16)  
                parsed = parsed.reshape(length,num_rows,num_cols) 
            elif(type == 2049):
                category = 'labels'
                parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=8) 
                parsed = parsed.reshape(length)                            
            if (length==10000):
                set = 'test'
            elif (length==60000):
                set = 'train'
            data_dict[set+'_'+category] = parsed 
        
        
        
  
                            
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bin_n=9
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:8,:8], bins[8:,:8], bins[:8,8:], bins[8:,8:]
    mag_cells = mag[:8,:8], mag[8:,:8], mag[:8,8:], mag[8:,8:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

h=1
for i in range (h):
    h=data_dict['train_images'][i]  
    p=np.pad(h, 2,mode='constant')
    for o in range (0,31,16):# loop for outer block
        for u in range (0,24,8):
            cellshist =p[o:o+16,u:u+16]
            t=hog(cellshist)
            
    



