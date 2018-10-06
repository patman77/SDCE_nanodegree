import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from draw_boxes import draw_boxes
from search_classify import search_windows
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import os, sys
sys.path.insert(0, os.path.abspath("..")) # __init__.py seems to be not enough
sys.path.insert(0, os.path.abspath("../Project4_Advanced_Lane_Finding_Project")) # __init__.py seems to be not enough
print(os.path.abspath(".."))

# import of lane finding imports
import matplotlib.pyplot as plt
#from PIL import ImageGrab # damn, no Linux support
import glob # for reading files from path
import numpy as np
from pathlib import Path # for is_file
import pickle

plt.interactive(True) # without this, plot won't become visible in pycharm

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from Project4_Advanced_Lane_Finding_Project.apply_sobel import abs_sobel_thresh
from Project4_Advanced_Lane_Finding_Project.color_and_gradient import pipeline
#from curve_pixels import measure_curvature_pixels
from Project4_Advanced_Lane_Finding_Project.direction_gradient import dir_threshold
from Project4_Advanced_Lane_Finding_Project.lane_histogram import hist
from Project4_Advanced_Lane_Finding_Project.magnitude_gradient import mag_thresh
from Project4_Advanced_Lane_Finding_Project.prev_poly import fit_poly, search_around_poly
from Project4_Advanced_Lane_Finding_Project.radius_curve import measure_curvature_real2
from Project4_Advanced_Lane_Finding_Project.rgb_to_hls import hls_select
from Project4_Advanced_Lane_Finding_Project.sliding_window import find_lane_pixels, fit_polynomial, fit_polynomial2
from Project4_Advanced_Lane_Finding_Project.sliding_window_template import window_mask,find_window_centroids
from Project4_Advanced_Lane_Finding_Project.undistort import cal_undistort
from Project4_Advanced_Lane_Finding_Project.warp import corners_unwarp_improved

# import of lane finding main
from Project4_Advanced_Lane_Finding_Project.main  import calibrate_camera
#execfile("Project4_Advanced_Lane_Finding_Project.main.py")
#from Project4_Advanced_Lane_Finding_Project.main  import backproject_measurement
#from Project4_Advanced_Lane_Finding_Project.main  import draw_curvature_and_position


# import of lane finding imports
from Project4_Advanced_Lane_Finding_Project.undistort  import cal_undistort

# Read in cars and notcars
images = glob.glob('./*vehicles_smallset/**/*.jpeg', recursive=True)
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 18  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 64  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 707]  # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

image = mpimg.imread('bbox-example-image.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(window_img)
mpimg.imsave('candidates.png', window_img)
print(' ')



