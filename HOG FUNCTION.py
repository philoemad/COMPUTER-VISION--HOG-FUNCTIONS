import os,codecs,numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle 
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
        
        
        
  

        
#hog implemntation begin of code
print(data_dict['test_labels'][1])                             
imgtest = data_dict['train_images'][10]
hog_gradient=np.array([[0,0,0,0,0,0,0,0,0]])#gradient of the hog 9 values og each cell initialization
image_hog_gradient=np.array([[0,0,0,0,0,0,0,0,0]])# final image 
x = np.array([[1, 0, -1]])
y= np.array([[-1], [0], [1]])
length=1#len(data_dict['train_images']) it should be length of data set but due to my pc is not able to run all the photo of dataset
for i in range(length):
    p=np.array(data_dict['test_images'][i])
    p=np.pad(p, 2,mode='constant')
    imgy=signal.convolve(p, y, mode='same',method='direct')
    imgx =signal.convolve(p, x, mode='same',method='direct')
    vertical_gradient_square = numpy.power(imgy, 2)
    horizontal_gradient_square = numpy.power(imgx, 2)
    sum_squares = horizontal_gradient_square + vertical_gradient_square
    grad_magnitude = numpy.sqrt(sum_squares)
    theta=numpy.arctan(imgy/imgx)# out is in radian
    grad_direction = numpy.rad2deg(theta) #convert out to degree
    grad_direction=numpy.nan_to_num(grad_direction)
    grad_direction = grad_direction+180 #mak it from 0 to 180 and convert (-)negative angles to postive and have it from 0 to 180
    rowrange=8
    colrange=8
    for o in range (0,31,16):# loop for outer block
        for p in range (0,24,8):  # loop for block shift 8 
            for k in range(0,15,8):# loop for move inside block with 8 cell
                for l in range(0,15,8): # same as up but for row cell
                    for i in range(rowrange): # loop for pixel 
                        for j in range(colrange):
                            i=i+k+o
                            j=j+l+p
                            if grad_direction[i][j]<10:
                                hog_gradient[0][0]=hog_gradient[0][0]+grad_magnitude[i][j]# less than 10 mid of first range then add the value to this half
                            elif grad_direction[i][j]<30:# less than the next half as in video so it will be make by equation and divid range by the result of this equation to all next value
                                ro1=numpy.abs(grad_direction[i][j]-10/20)
                                ro2=numpy.abs(grad_direction[i][j]-30/20)
                                r1=grad_magnitude[i][j]*ro1
                                hog_gradient[0][0]=hog_gradient[0][0]+r1
                                r2=grad_magnitude[i][j]*ro1
                                hog_gradient[0][1]=hog_gradient[0][1]+r2
                            elif grad_direction[i][j]<50:
                                ro1=numpy.abs(grad_direction[i][j]-30/20)
                                ro2=numpy.abs(grad_direction[i][j]-50/20)
                                r1=grad_magnitude[i][j]*ro1
                                hog_gradient[0][1]=hog_gradient[0][1]+r1
                                r2=grad_magnitude[i][j]*ro1
                                hog_gradient[0][2]=hog_gradient[0][2]+r2
                            elif grad_direction[i][j]<70:
                                ro1=numpy.abs(grad_direction[i][j]-50/20)
                                ro2=numpy.abs(grad_direction[i][j]-70/20)
                                r1=grad_magnitude[i][j]*ro1
                                hog_gradient[0][2]=hog_gradient[0][2]+r1
                                r2=grad_magnitude[i][j]*ro1
                                hog_gradient[0][3]=hog_gradient[0][3]+r2
                            elif grad_direction[i][j]<90:
                                ro1=numpy.abs(grad_direction[i][j]-70/20)
                                ro2=numpy.abs(grad_direction[i][j]-90/20)
                                r1=grad_magnitude[i][j]*ro1
                                hog_gradient[0][3]=hog_gradient[0][3]+r1
                                r2=grad_magnitude[i][j]*ro1
                                hog_gradient[0][4]=hog_gradient[0][4]+r2
                            elif grad_direction[i][j]<110:
                                ro1=numpy.abs(grad_direction[i][j]-90/20)
                                ro2=numpy.abs(grad_direction[i][j]-110/20)
                                r1=grad_magnitude[i][j]*ro1
                                hog_gradient[0][4]=hog_gradient[0][4]+r1
                                r2=grad_magnitude[i][j]*ro1
                                hog_gradient[0][5]=hog_gradient[0][5]+r2
                            elif grad_direction[i][j]<130:
                                 ro1=numpy.abs(grad_direction[i][j]-110/20)
                                 ro2=numpy.abs(grad_direction[i][j]-130/20)
                                 r1=grad_magnitude[i][j]*ro1
                                 hog_gradient[0][5]=hog_gradient[0][5]+r1
                                 r2=grad_magnitude[i][j]*ro1
                                 hog_gradient[0][6]=hog_gradient[0][6]+r2
                            elif grad_direction[i][j]<150:
                                ro1=numpy.abs(grad_direction[i][j]-130/20)
                                ro2=numpy.abs(grad_direction[i][j]-150/20)
                                r1=grad_magnitude[i][j]*ro1
                                hog_gradient[0][6]=hog_gradient[0][6]+r1
                                r2=grad_magnitude[i][j]*ro1
                                hog_gradient[0][7]=hog_gradient[0][7]+r2
                            elif grad_direction[i][j]<170:
                                 ro1=numpy.abs(grad_direction[i][j]-150/20)
                                 ro2=numpy.abs(grad_direction[i][j]-170/20)
                                 r1=grad_magnitude[i][j]*ro1
                                 hog_gradient[0][7]=hog_gradient[0][7]+r1
                                 r2=grad_magnitude[i][j]*ro1
                                 hog_gradient[0][8]=hog_gradient[0][8]+r2
                            i=0#to free the loop of the cells
                            j=0   
                    image_hog_gradient=np.concatenate((image_hog_gradient,hog_gradient),axis=1)
                    hog_gradient=np.array([[0,0,0,0,0,0,0,0,0]])        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    









index = [0,1,2, 3,4,5, 6,7,8]
image_hog_gradient=np.delete(image_hog_gradient, index)# as i but the inilization with 9 zeros so i remove them insiatize in line 49
image_hog_gradient=np.reshape(image_hog_gradient,(1,216)) #as the out of 32 pixel image will be 6 blocks ecach has 32 value and it will be diffrent from image size to anther 
                               

# Load the model from the file 
svm_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
u=svm_from_joblib.predict(image_hog_gradient) 
print(u)                             