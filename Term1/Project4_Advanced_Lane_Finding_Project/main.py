import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
#from PIL import ImageGrab # damn, no Linux support
import glob # for reading files from path
import numpy as np
from pathlib import Path # for is_file
import pickle

plt.interactive(True) # without this, plot won't become visible in pycharm

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
nx = 9
ny = 6
calibimgpath = './camera_cal/calibration*.jpg'
calibfilename = 'calib.p'
#some params for plots
horiz_spacing = .2
width_space = .05

def process_image(image):
    pass


#image_street = mpimg.imread('test_images/test1.jpg ')
#image = mpimg.imread('camera_cal/calibration2.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# start the pipeline
# step 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

def calibrate_camera(img_path, xnum=nx, ynum=ny, recalc=False, calib_filename='calib.p'):
    if recalc is False:
        print("recalc false")
        # check if there is a saved calib file
        if Path(calib_filename).is_file():
            print("Found existing calibration file under the given name", calib_filename, "returning")
            return
        else:
            print("No existing calibration file under the given name", calib_filename, "continuing")


    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:xnum, 0:ynum].T.reshape(-1, 2)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []
    image_list = glob.glob(calibimgpath)
    fig, axs = plt.subplots(4, 5, figsize=(15,10))
    fig.subplots_adjust(hspace = .3, wspace = width_space)
    axs = axs.ravel()
    for i, filename in enumerate(image_list):
        print("reading img", i, "from", len(image_list))
        image = cv2.imread(filename)
        if image is None:
            continue
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            # refine img points, take from (C++) opencv documentation:
            # https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(image, (nx, ny), corners_refined, ret)
            axs[i].axis('off')
            axs[i].imshow(image)
            plt.show(block=False)
            cv2.waitKey(500)
        else: # also display image when corners are not found, but mark
            height, width, channels = image.shape
            cv2.line(image, (0, 0), (width, height), (255, 0, 0), 25)
            cv2.line(image, (width, 0), (0, height), (255, 0, 0), 25)
            axs[i].axis('on')
            axs[i].imshow(image)
    print('waiting for key')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # now calibrate
    print('calibrating...')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('calibrated, saving...')
    dist_pickle = {}
    dist_pickle["mtx"]  = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(calib_filename, "wb"))
    return mtx, dist

# From chapter 3. Tips and Tricks for the Project:
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

# -----------------------------------------------------------------------------
# ----------------------------------start of main -----------------------------
# -----------------------------------------------------------------------------

try:
    if Path(calibfilename).is_file():
        print("Found existing calibration file under the given name", calibfilename, "using that one")
        dist_pickle = pickle.load(open(calibfilename, "rb"))
        mtx  = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

    else:
        print("No existing calibration file under the given name", calibfilename, "recalculating")
        mtx, dist = calibrate_camera(calibimgpath)
except:
    print('exception!!')
    exit(1)

# test undistortion
# step 2: Apply a distortion correction to raw images.
testimg = mpimg.imread('./camera_cal/calibration1.jpg')
#testimg = cv2.imread('./test_images/test2.jpg')
dst = cv2.undistort(testimg, mtx, dist, None, mtx)
#undistorted = cal_undistort(image, objpoints, imgpoints)

# visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
f.subplots_adjust(hspace = horiz_spacing, wspace=width_space)
ax1.set_title("original image")
ax1.axis('off')
ax1.imshow(testimg)
ax2.set_title("undistorted image")
ax2.axis('off')
ax2.imshow(dst)
print(" ")
#cv2.waitKey(10000000)

# step 3: Use color transforms, gradients, etc., to create a thresholded binary image.

# step 4: Apply a perspective transform to rectify binary image ("birds-eye view").

# step 5: Detect lane pixels and fit to find the lane boundary.

# step 6: Determine the curvature of the lane and vehicle position with respect to center.

# step 7: Warp the detected lane boundaries back onto the original image.

# step 8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.