import cv2
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from utils import *
from line import Line

# path to videos and results
video_file = './project_video.mp4'
ouput_file = './result_project_video.mp4'

# parameters for camera calibration
path_cal = './camera_cal/*.jpg'
nx,ny = 9,6
test_image = './camera_cal/calibration3.jpg'
objPoints, imgPoints, cornerImgs = mapPoints(path_cal, nx, ny) # camera calibration is done globally
raw_img = cv2.imread(test_image)
_, _, mtx, dist = calUndistort(raw_img, objPoints, imgPoints) # get the transforming matrix for image undistortion

# parameters for image filtering
sobel_kernel = 9 # sobel kernel size
s_thresh = (20,230) # sobel filter thresholds
m_thresh = (30,220) # magnitude of gradient thresholds
d_thresh = (0.7,1.2) # direction of gradient threhsolds
cs_thresh = (90, 255) # color threshold for S channel in HLS space
ch_thresh = (15, 100) # color threshold for H channel in HLS space

# paramters for perspective transform. Following source points and target points were obtained from given images with straight lane lines
srcPoints = np.array([[213,700],[592,450],[687,450],[1090,700]])
tgtPoints = np.array([[300,700],[300,20],[900,20],[900,700]])

# parameters for lane searching and curvature radius calculation
nwindows = 9
margin = 100
minpix = 50
ym_per_pix = 30.0/720.0
xm_per_pix = 3.7/600.0

# running parameters
scanned = False # identify if a previous run has been done to scan the location of lanes
left_lane = Line(n=10) # define instances for the left and right lanes. The smoothing window is 10
right_lane = Line(n=10)
left_fit = [] # lists of coefficients of the latest fitted (smoothed in the window) polynomial
right_fit = []


def img_annotation(raw_img):
    '''
    run the pipeline and return an annotated image img with detected lanes and curvature radius on it
    '''
    # global parameters for the pipeline
    global mtx, dist
    global sobel_kernel, s_thresh, m_thresh, d_thresh, cs_thresh, ch_thresh
    global srcPoints, tgtPoints
    global nwindows, margin, minpix, ym_per_pix, xm_per_pix
    global scanned, left_lane, right_lane, left_fit, right_fit
    
    # undistort the image
    undst_img = cv2.undistort(raw_img, mtx, dist, None, mtx)
    # apply thresholds
    comb_img =  img_thresh(undst_img, s_thresh, m_thresh, d_thresh, cs_thresh, ch_thresh, sobel_kernel)
    # perspetive transform
    warped_comb_img, M, Minv = img_warper(comb_img, np.float32(srcPoints), np.float32(tgtPoints))
    
    # check if a previous image has been scanned
    if scanned == False: # if there is no image has been scanned before, run the sliding window from scratch,and 
        # I also chose to run a 2nd round of search within the polynomial zones of the results of the 1st round search at here
        _,left_fit,right_fit = findLines(warped_comb_img, nwindows, margin, minpix)
        
        out_img, window_img, leftxy, rightxy, left_fit, right_fit = findLines_poly(warped_comb_img, left_fit, right_fit, margin)
        
        ploty, left_fitx, right_fitx, left_curverad, right_curverad, y_eval,left_fit_cr,right_fit_cr = laneFit(leftxy, rightxy, out_img, ym_per_pix, xm_per_pix)
        scanned = True
        
    else: # if there are a pair of polynomials from previous images, use the faster search function
        out_img, window_img, leftxy, rightxy, left_fit, right_fit = findLines_poly(warped_comb_img, left_fit, right_fit, margin)
        
        ploty, left_fitx, right_fitx, left_curverad, right_curverad, _ , _, _ = laneFit(leftxy, rightxy, out_img, ym_per_pix, xm_per_pix)
       
    # add the new polynomials to the list
    left_fit = left_lane.add(left_fit)
    right_fit = right_lane.add(right_fit)
    
    # convert the fitted polynomials to the raw image
    undst_lane_img = invLanes(undst_img, warped_comb_img, ploty, left_fitx, right_fitx, Minv)
    
    # calculate car offset
    offset = vehOffset(undst_lane_img, left_fit, right_fit, xm_per_pix)
    
    # add curvature radius on to the image
    img = addText(undst_lane_img, left_curverad, right_curverad, offset)
    return img

if __name__ == '__main__':
    # process the video clip
    video = VideoFileClip(video_file)
    annotated_video = video.fl_image(img_annotation)
    annotated_video.write_videofile(ouput_file, audio=False)

   