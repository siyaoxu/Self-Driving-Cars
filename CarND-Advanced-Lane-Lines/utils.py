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

def BGR2GRAY(raw_img):
    '''
    Convert a BGR image to Grayscale
    '''
    return cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

def sobel_thresh(gray, orient='x', thresh=(10,245), sobel_kernel = 3):
    '''
    Take the derivative of a grayscale image gray in x or y direction
    Take the absolute value of the derivative and threshold then with thresh_min and thresh_max
    create and return a mask of 1's where the scaled gradient magnitude is >thresh_min and < thresh_max
    '''
    if orient == 'x':
        O = [1,0]
    else:
        O = [0,1]
    sobel = cv2.Sobel(gray, cv2.CV_64F, O[0], O[1], sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary, sobel

def sobel_mag_thresh(sobelx, sobely, mag_thresh=(10, 245)):
    '''
    Take the gradient in x and y seperately
    Calculate the magnitude of the gradients
    Scale to 8-bit (0-255) and convert to type np.uint8
    Create a binary mask where mag_thresh are met
    Return the mask as a binary image
    '''
#    from IPython.core.debugger import Tracer; Tracer()()
    abs_sobelxy = np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary

def dir_thresh(sobelx, sobely, thresh=(0.2, 1.3)):
    '''
    Take the gradient of an grayscale image gray in x and y directions
    Take the absolute value of x and y gradients
    Calculate the direction of the gradient
    Create a binary mask where direction thresholds are met
    return the mask
    '''
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    binary = np.zeros_like(dir_grad)
    binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return binary

def grad_thresh(raw_img, s_thresh=(10, 245), m_thresh=(10, 245), d_thresh=(0.2, 1.3), sobel_kernel = 3):
    '''
    Take a raw BGR image and convert it to grayscale
    Calculate thresholds based on x and y derivatives, magnitude of gradients, and direction of gradients
    Combine masks of three thresholds and export a binary mask
    '''
    gray = BGR2GRAY(raw_img)
    bi_sobelx, sobelx = sobel_thresh(gray, 'x', s_thresh, sobel_kernel)
    bi_sobely, sobely = sobel_thresh(gray, 'y', s_thresh, sobel_kernel)
    bi_mag_grad = sobel_mag_thresh(sobelx, sobely, m_thresh)
    bi_dir_grad = dir_thresh(sobelx, sobely, d_thresh)
    
    binary = np.zeros_like(gray)
    binary[((bi_sobelx == 1) & (bi_sobely == 1)) | ((bi_mag_grad == 1) & (bi_dir_grad == 1))] = 1
    return binary

def hls_thresh(raw_img, s_thresh=(10,255), h_thresh=(10,255)):
    '''
    Convert a BGR image raw_img to HLS space
    Apply a threshold to the S channel
    Return a binary image of threshold result
    '''
    hls = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HLS)
    s_img = hls[:,:,2]
    h_img = hls[:,:,0]
    binary = np.zeros_like(s_img)
    binary[(s_img > s_thresh[0]) & (s_img <= s_thresh[1]) & (h_img > h_thresh[0]) & (h_img <= h_thresh[1])] = 1
    return binary

def img_thresh(raw_img, s_thresh=(10, 245), m_thresh=(10, 245), d_thresh=(0.2, 1.3), cs_thresh = (10,255), ch_thresh = (15, 100), sobel_kernel = 3):
    '''
    Combine gradient thresholds and color thresholds
    '''
    bi_grad = grad_thresh(raw_img, s_thresh, m_thresh, d_thresh, sobel_kernel)
    bi_hls_s = hls_thresh(raw_img, cs_thresh,ch_thresh)
    binary = np.zeros_like(bi_grad)
    binary[(bi_grad == 1) | (bi_hls_s == 1)] = 1
    return binary

def img_roi(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def img_warper(raw_img, src, tgt):
    '''
    perform perspective convertion on raw_img using points src, tgt
    generate an output topview of the image
    '''
    src = np.float32(src)
    tgt = np.float32(tgt)
    M = cv2.getPerspectiveTransform(src, tgt)
    Minv = cv2.getPerspectiveTransform(tgt, src)
    warped_img = cv2.warpPerspective(raw_img, M, (raw_img.shape[1], raw_img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped_img, M, Minv

def findLines(binary_warped, nwindows = 9, margin = 100, minpix = 50):
    '''
    Detect the left and right lanes of a warped binary mask
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

#     # Choose the number of sliding windows
#     nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
#     # Set the width of the windows +/- margin
#     margin = 100
#     # Set minimum number of pixels found to recenter window
#     minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return out_img, left_fit, right_fit

def findLines_poly(binary_warped, left_fit, right_fit, margin = 100):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    window_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return out_img, window_img, zip(leftx,lefty), zip(rightx,righty), left_fit, right_fit

def laneFit(leftPoints, rightPoints, binary_warped, ym_per_pix=1, xm_per_pix=1):
    '''
    Fit a 2nd order polynomial to the left lane and right lane points seperately
    and calculate the curvatures for both lane lines
    '''
    leftxy = np.array(list(leftPoints))
    rightxy = np.array(list(rightPoints))
   
    left_fit = np.polyfit(leftxy[:,1], leftxy[:,0], 2)
    right_fit = np.polyfit(rightxy[:,1], rightxy[:,0], 2)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(leftxy[:,1]*ym_per_pix, leftxy[:,0]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(rightxy[:,1]*ym_per_pix, rightxy[:,0]*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return ploty, left_fitx, right_fitx, left_curverad, right_curverad, y_eval,left_fit_cr[0:2],right_fit_cr[0:2]

def invLanes(undist_img, warped, ploty, left_fitx, right_fitx, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_img.shape[1], undist_img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
    return result

def vehOffset(undist, left_fit, right_fit, xm_per_pix):
    """
    Calculate vehicle offset
    """
    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    offset *= xm_per_pix

    return offset

def addText(img, left_curverad, right_curverad, offset):
    """
    Annotate the image with averaged curvature radius of two lanes
    """
    # Average the curvature radius of two lanes
    avg_curverad = (left_curverad + right_curverad)/2
    annotation = 'Radius of curvature: {:5.1f} m        Vehicle offset from the center: {:5.1f} m'.format(avg_curverad, offset)
    result = cv2.putText(img, annotation, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    return result