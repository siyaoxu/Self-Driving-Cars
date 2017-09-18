# functions used for the lane detection pipeline

import numpy as np
import glob
import cv2

def findCorners(raw_img, nx, ny):
    '''
    Find corner points of a chessboard image
    raw_img: the raw chessboard image (BGR)
    nx, ny: columns, rows of the image
    return:
    retval, corners, corner_img
    '''
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(gray_img, (nx,ny), None)
    
    corner_img = np.empty(shape = raw_img.shape[::-1])
    if retval:
        corner_img = raw_img.copy()
        cv2.drawChessboardCorners(corner_img,(nx,ny),corners,retval)
    return retval, corners, corner_img

def objP(nx,ny):
    '''
    generate cordinates for points in 3D
    '''
    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    return objp

def mapPoints(path_cal_img, nx, ny):
    '''
    Map points in calibration images to points in the 3D space
    path_cal_img: the path to the folder of calibration images
    '''
    objp = objP(nx,ny)
    
    objPoints = [] # cordinates in 3D space
    imgPoints = [] # cordinates in 2D images
    cornerImgs = [] # store images with corner points
    
    images = glob.glob(path_cal_img) # retrieve a list of paths to calibration images
    
    for idx, fname in enumerate(images):
        # read the image
        raw_img = cv2.imread(fname)
#        from IPython.core.debugger import Tracer; Tracer()()
        # find corners 
        retval, corners, corner_img = findCorners(raw_img, nx, ny)
        
        # if corners were found, map them using two lists
        if retval:
            objPoints.append(objp)
            imgPoints.append(corners)
            cornerImgs.append(corner_img)
    
    return objPoints, imgPoints, cornerImgs

def calUndistort(raw_img, objPoints, imgPoints):
    '''
    Undistort 
    raw_img: the raw image to be undistorted
    objPoints, imgPoints: points in the 3D space, points in 2D
    '''
    retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, raw_img.shape[::-1][1:], None, None)
    undst_img = cv2.undistort(raw_img, mtx, dist, None, mtx)
    
    return retval, undst_img, mtx, dist
    