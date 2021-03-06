import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    #print(corners.shape())
    # 4) If corners found: 
            # a) draw corners
    dst = cv2.drawChessboardCorners(dst, (8,6), corners, ret)
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    srcpts = np.float32([ [corners[ 0][0][0],corners[ 0][0][1] ],
                          [corners[ 7][0][0],corners[ 7][0][1] ],
                          [corners[ 47][0][0],corners[ 47][0][1] ],
                          [corners[ 40][0][0],corners[ 40][0][1] ] ] )
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dstpts = np.float32([[80.0, 81.666666666666666],
                         [120.0, 81.666666666666666],
                         [120.0, 878.33333333333333],
                         [80.0, 878.33333333333333]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(srcpts, dstpts)
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    #delete the next two lines
    #M = None
    #warped = np.copy(dst) 
    warped = cv2.warpPerspective(dst, M, (1280,960))
    return warped, M

def corners_unwarp_improved(img, nx, ny, mtx, dist):

    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    srcpts = np.float32([[600.0, 450.0],
                         [686.0, 450.0],
                         [1040.0, 680.0],
                         [270.0, 680.0]])
                 #Note: you could pick any four of the detected corners
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dstpts = np.float32([[270.0, 0.0],
                         [1040.0, 0.0],
                         [1040.0, 720.0],
                         [270.0, 720.0]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(srcpts, dstpts)
    Minv = cv2.getPerspectiveTransform(dstpts, srcpts)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (1280,720))
    return warped, M, Minv

# # Read in the saved camera matrix and distortion coefficients
# # These are the arrays you calculated using cv2.calibrateCamera()
# dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
# mtx = dist_pickle["mtx"]
# dist = dist_pickle["dist"]
#
# # Read in an image
# img = cv2.imread('test_image2.png')
# nx = 8 # the number of inside corners in x
# ny = 6 # the number of inside corners in y
#
#
# top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(top_down)
# ax2.set_title('Undistorted and Warped Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

