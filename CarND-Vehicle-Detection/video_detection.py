import cv2
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from heat import Heatmap
from lane_detection_utils import *
from vehicle_detection_utils import *
from line import Line
import pickle
import warnings

# path to videos and results
# video_file = './test_video.mp4'
# ouput_file = './result_test_video.mp4'

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

# parameters for vehicle detection
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
# y_start_stops = [[400, 500],[450, 550],[500, 650],[620, None]] # Min and max in y to search in slide_window()
# xy_windows = [(70,70),(110,110),(150,150),(190,190),(230,230)]
# xy_overlap = (0.8,0.8)
y_start_stops = [[400, 500],[420, 550],[450, 600],[480, None]] # Min and max in y to search in slide_window()
xy_windows = [(60,60),(70,70),(100,100),(130,130)]
xy_overlap = (0.7,0.7)

# load the svc classifier
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    models = pickle.load( open("svc-clf.sav", "rb" ) )
    
clf = models[0].best_estimator_
X_scaler = models[1]
cspace = models[2]
spatial_size = models[3]
hist_bins = models[4]
hist_range = models[5]
orient = models[6]
pix_per_cell = models[7]
cell_per_block = models[8]
hog_channel = models[9]
heat_thresh = 2.0
# using a Heatmap class to keep track of detected objects in the past n frames 
heatHistory = Heatmap(n = 30, shape = [720,1280])

def vehDetect(image,lane_image, y_start_stops, xy_windows, xy_overlap, 
                     clf, X_scaler, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, 
                     spatial_feat = True, hist_feat = True, hog_feat = True, heat_thresh = 1, x_start_stop=[None, None]):
    '''
    the vehicle detection pipeline is implemented in this function
    '''
    global heatHistory
    multi_hot_windows = []

    for xy_window,y_start_stop in zip(xy_windows,y_start_stops):
        windows = slide_window(image, x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=xy_window, xy_overlap=xy_overlap)

        hot_windows = search_windows(image, windows, clf, X_scaler, color_space=cspace, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        multi_hot_windows+=hot_windows
    
    window_img = draw_boxes(image.copy(), multi_hot_windows, color=(255, 0, 0), thick=6)                    
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,multi_hot_windows)
    
    avgHeat = heatHistory.add(heat)

    # Apply threshold to help remove false positives
    avgHeat = apply_threshold(avgHeat,heat_thresh)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(avgHeat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(lane_image), labels)
    
    return windows, hot_windows, window_img, avgHeat, heatmap, labels, draw_img

def img_annotation(raw_img):
    '''
    run the pipeline and return an annotated image img with detected lanes, curvature radius, and the detected vehicles on it
    '''
    # global parameters for the pipeline
    global mtx, dist
    global sobel_kernel, s_thresh, m_thresh, d_thresh, cs_thresh, ch_thresh
    global srcPoints, tgtPoints
    global nwindows, margin, minpix, ym_per_pix, xm_per_pix
    global scanned, left_lane, right_lane, left_fit, right_fit
    
    global spatial_feat, hist_feat, hog_feat, y_start_stops, xy_windows, xy_overlap
    global clf, X_scaler, cspace, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, heat_thresh
    
    global heatHistory, avgHeat
    
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
    
    # detect vehicles on the undistorted image
    windows, hot_windows, window_img, avgHeat, heatmap, labels, draw_img = vehDetect(image = undst_img, lane_image = img, 
                         y_start_stops = y_start_stops, xy_windows = xy_windows, xy_overlap = xy_overlap, 
                         clf = clf, X_scaler = X_scaler, cspace = cspace, spatial_size = spatial_size, hist_bins = hist_bins, 
                         orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, hog_channel = hog_channel, 
                         spatial_feat = True, hist_feat = True, hog_feat = True, heat_thresh = heat_thresh, x_start_stop=[None, None])
    
    return draw_img

if __name__ == '__main__':
    # process the video clip
    video = VideoFileClip(video_file)
    annotated_video = video.fl_image(img_annotation)
    annotated_video.write_videofile(ouput_file, audio=False)

   