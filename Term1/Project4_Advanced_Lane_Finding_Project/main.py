import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import glob # for reading files from path

# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
from IPython.display import HTML

from apply_sobel import abs_sobel_thresh
from color_and_gradient import pipeline
from curve_pixels import measure_curvature_pixels
from direction_gradient import dir_threshold
from lane_histogram import hist
from magnitude_gradient import mag_thresh
from prev_poly import fit_poly, search_around_poly
from radius_curve import measure_curvature_real
from rgb_to_hls import hls_select
from sliding_window import find_lane_pixels, fit_polynomial
from undistort import cal_undistort
from warp import corners_unwarp
#from findchessboardcorners import nx, ny
nx = 8
ny = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def process_image(image):
    pass


image_street = mpimg.imread('test_images/test1.jpg')
image = mpimg.imread('camera_cal/calibration2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# start the pipeline
# step 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

def calibrate_camera(img_path):
    image_list = []
    for filename in glob.glob(img_path):
        image = mpimg.imread(filename)
        if image != 0:
            image_list.append(image)
        ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)
    return image_list


ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
    plt.imshow(image)
#objpoints =
#imgpoints =
#undistorted = cal_undistort(image, objpoints, imgpoints)
# step 2: Apply a distortion correction to raw images.

# step 3: Use color transforms, gradients, etc., to create a thresholded binary image.

# step 4: Apply a perspective transform to rectify binary image ("birds-eye view").

# step 5: Detect lane pixels and fit to find the lane boundary.

# step 6: Determine the curvature of the lane and vehicle position with respect to center.

# step 7: Warp the detected lane boundaries back onto the original image.

# step 8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.