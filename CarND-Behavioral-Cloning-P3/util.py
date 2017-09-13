# utilities
import csv
import cv2
import h5py
import pickle
import numpy as np
from random import shuffle

def resizeImg(img,small = (200,128)):
    '''
    resize the image to 200x128
    '''
    return cv2.resize(img,small)

def cropImg(img, keep = [46,110]):
    '''
    crop the image to the road zone
    '''
    croppedImg = img[keep[0]:keep[1],:,:]
    return croppedImg

def flipImgRnd(img, measurement, t):
    '''
    randomly decide if the image was going to be flipped
    '''
    test = np.random.rand()
    if  test < t:
        measurement = -measurement
        img = cv2.flip(img,1)
    return img, measurement

def tranX(img, measurement, ftxy, f_steering):
    '''
    randomly shift the image horizontally with a steering correction factor f_steering
    '''
    nrow,ncol,_ = img.shape
    # translate the rotated image
    M = np.float32([[1, 0, np.float32(ncol)*ftxy[0]], [0, 1 , np.float32(nrow)*ftxy[1]]])
    shifted_img = cv2.warpAffine(img,M,(ncol,nrow))
    # adjust the measurement
#    measurement = measurement+np.sign(ftxy[0])*np.sqrt(ftxy[0]**2 + ftxy[1]**2)*f_steering
    measurement = measurement+ftxy[0]*f_steering
    return shifted_img, measurement

def brightness(hsv_img, fv):
    '''
    change the brightness of the image
    '''
#     from IPython.core.debugger import Tracer
#     Tracer()()
    hsv_img[:,:,2] = hsv_img[:,:,2]*fv
    hsv_img[:,:,2][hsv_img[:,:,2]>255.0] = 255.0
    b_img = np.array(hsv_img,dtype=np.uint8)
    return b_img

def gausNoise(hsv_img, ksize = (21,21)):
    '''
    Generate random shadow to the image by adding a Gaussian blurred random noise to the V channel
    '''
    nrow,ncol,_ = hsv_img.shape
    shadow = cv2.GaussianBlur(np.random.rand(nrow, ncol),ksize,0)
    hsv_img[:,:,2] = hsv_img[:,:,2]*shadow
    hsv_img[:,:,2] = (hsv_img[:,:,2] - np.amin(hsv_img[:,:,2]))/(np.amax(hsv_img[:,:,2])-np.amin(hsv_img[:,:,2]))*255.0
    shadow_img = np.array(hsv_img,dtype=np.uint8)
    return shadow_img

def procImg(raw_img):
    # convert the image to hsv domain, and will train the model in this domain
    hsv_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2HSV)

    # resize and crop the raw image
    img = cropImg(resizeImg(hsv_img))
    return img

# define a funciton to sample values for parameters of the image augmentation function
def AugmParams(rfbrightness, rftx, rfty, size = 1 ):
    fbrightness = np.float32(np.random.uniform(rfbrightness[0],rfbrightness[1]))
    ftxys = list(zip(np.float32(np.random.uniform(rftx[0],rftx[1],size)),np.float32(np.random.uniform(rfty[0],rfty[1],size))))
    return fbrightness, ftxys

# process the original image to get one sample to train/validate the model
def augmImg(img, measurement):
    
    # randomly decide if the image was going to be flipped
    flipped_img, measurement = flipImgRnd(img, measurement,0.5)

    
    # get a set of parameters for image augmentation
    fv, ftxy = AugmParams(rfbrightness = (0.3,2.5), rftx = (-0.5, 0.5), rfty = (-0.5, 0.5), size = 1)
    
    # translation
    shifted_img, measurement = tranX(flipped_img, measurement, ftxy[0], f_steering = 0.5)
#    shadow_img = shifted_img

    # gaussian noise
    shadow_img = gausNoise(shifted_img)

    # adjust the brightness
    b_img = brightness(shadow_img, fv)
    
#    # HSV2YUV
#    b_img = cv2.cvtColor(cv2.cvtColor(b_img, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2YUV)
    
    return b_img, measurement