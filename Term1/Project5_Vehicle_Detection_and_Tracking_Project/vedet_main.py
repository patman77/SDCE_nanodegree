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
from heatmap import add_heat
from heatmap import apply_threshold
from heatmap import draw_labeled_bboxes
from hog_subsample import find_cars
from hog_subsample import find_cars2
from scipy.ndimage.measurements import label


from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import os, sys
sys.path.insert(0, os.path.abspath("..")) # __init__.py seems to be not enough
sys.path.insert(0, os.path.abspath("../Project4_Advanced_Lane_Finding_Project")) # __init__.py seems to be not enough
print(os.path.abspath(".."))

print("Current working dir ", os.getcwd())
os.chdir("../Project4_Advanced_Lane_Finding_Project")
print("Current working dir ", os.getcwd())

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
#execfile("Project4_Advanced_Lane_Finding_Project.main.py") # bad idea
from Project4_Advanced_Lane_Finding_Project.main  import backproject_measurement
from Project4_Advanced_Lane_Finding_Project.main  import draw_curvature_and_position
from Project4_Advanced_Lane_Finding_Project.main import process_image_lane_detect


# import of lane finding imports
from Project4_Advanced_Lane_Finding_Project.undistort  import cal_undistort

# change back
os.chdir("../Project5_Vehicle_Detection_and_Tracking_Project")

# Class to have a heatmap memory

class Rectangle_Memory:
    def __init__(self, historysize=15):
        # history of rectangles
        self.previous_rects = []
        self.historysize = historysize

    def add_rect(self, rectangles):
        self.previous_rects.append(rectangles)
        if len(self.previous_rects)>self.historysize:
            # delete oldest rectangles
            self.previous_rects = self.previous_rects[len(self.previous_rects)-self.historysize:]


def process_image_vedet(image):
    dst = process_image_lane_detect(image)
    draw_image = np.copy(dst)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.8, 0.8))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    #hot_windows = [item for sublist in hot_windows for item in sublist]
    # add hot windows to history
    if len(hot_windows)>0:
        memory.add_rect(hot_windows)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 255, 255), thick=1)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    for rect in memory.previous_rects:
        heat = add_heat(heat, rect)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1 + len(memory.previous_rects)//2)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(window_img, labels)
    return draw_img

modelfilename = "model.svc"
### TODO: Tweak these parameters and see how the results change.
#color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32
# Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 707]  # Min and max in y to search in slide_window()
y_start_stop_single1_of_2 = [y_start_stop[0]  , int((y_start_stop[0]+y_start_stop[1])/2) ]
y_start_stop_single2_of_2 = [int((y_start_stop[0]+y_start_stop[1])/2)+1, y_start_stop[1] ]

# check if previously saved model exists, then use it
if os.path.isfile(modelfilename):
    with open(modelfilename, mode="rb") as f:
        data = pickle.load(f)
    if "svc" in data:
        svc = data["svc"]
        X_scaler = data["scaler"]
        print("Loaded linear SVC classifier:")
        car_features = data["cars_features"]
        notcar_features = data["notcars_features"]
        color_space = data["colorspace"]
        orient = data["orient"]
        pix_per_cell = data["pix_per_cell"]
        cell_per_block = data["cell_per_block"]
        hog_channel = data["hog_channel"]
        spatial_size = data["spatial_size"]
        hist_bins = data["hist_bins"]
        spatial_feat = data["spatial_feat"]
        hist_feat = data["hist_feat"]
        hog_feat = data["hog_feat"]

        # # Create an array stack of feature vectors
        # X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # # Define the labels vector
        # y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        # X_scaler = StandardScaler().fit(X)

    else:
        svc = None
        print("Loaded linear SVC classifier:")

    print("Loaded model!")
else:
    svc = None
    # Read in cars and notcars
    carimages = glob.glob('./vehicles/**/*.png', recursive=True)
    notcarimages = glob.glob('./non-vehicles/**/*.png', recursive=True)
    cars = []
    notcars = []
    for image in carimages:
        cars.append(image)
    for image in notcarimages:
        notcars.append(image)

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = min(len(cars), len(notcars))
    print('#cars = ', len(cars), ' #notcars = ', len(notcars), 'sample_size = ', sample_size)
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    fig.subplots_adjust(hspace=.1, wspace=.05)
    axs = axs.ravel()

    # Plot some car and non-car images
    for i in np.arange(32):
        img = cv2.imread(carimages[np.random.randint(0, len(carimages))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('car', fontsize=10)
        axs[i].imshow(img)
    for i in np.arange(32, 64):
        img = cv2.imread(notcarimages[np.random.randint(0, len(notcarimages))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('no car', fontsize=10)
        axs[i].imshow(img)
   # mpimg.imsave('report_images/figure01.png', img)
    print(" ")

    # Plot some HOG images
    car_img = mpimg.imread(carimages[0])
    hog_channel = 0
    _, car_HOG = get_hog_features(car_img[:, :, hog_channel], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    noncar_img = mpimg.imread(notcarimages[0])
    _, notcar_HOG = get_hog_features(noncar_img[:, :, hog_channel], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)

    # Visualize
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
    axs.ravel()
    fig2.subplots_adjust(hspace=.4, wspace=.2)
    ax1.imshow(car_img)
    ax1.set_title('Car Image', fontsize=16)
    ax2.imshow(car_HOG, cmap='gray')
    ax2.set_title('Car HOG', fontsize=16)
    ax3.imshow(noncar_img)
    ax3.set_title('Not-Car Image', fontsize=16)
    ax4.imshow(notcar_HOG, cmap='gray')
    ax4.set_title('Not-Car HOG', fontsize=16)
    print('...')


# load or train from scratch
    if svc is None: # train
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

        # Use a linear SVC
        svc = LinearSVC()

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


        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
    else: # load the last training result
        print("use the loaded svc together with training")


    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    data = {
        "svc": svc,
        "scaler": X_scaler,
        "cars_features": car_features,
        "notcars_features": notcar_features,
        "colorspace": color_space,
        "orient": orient,
        "pix_per_cell": pix_per_cell,
        "cell_per_block": cell_per_block,
        "hog_channel": hog_channel,
        "spatial_size": spatial_size,
        "hist_bins": hist_bins,
        "spatial_feat": spatial_feat,
        "hist_feat": hist_feat,
        "hog_feat": hog_feat
    }
    # Save the training and test features:
    with open(modelfilename, mode="wb") as f:
        pickle.dump(data, f)



doitonthevideo = True
#doitonthevideo = False
historysize = 15
memory = Rectangle_Memory(historysize)

if doitonthevideo == False:
    testimages = glob.glob('./test_images/*.jpg', recursive=True)
    #testimages = glob.glob('./video_single_images/0000*.jpg', recursive=True)
    outimgpath = './output_images/'

    for filename in testimages:
        image = mpimg.imread(filename)
        if image is None:
            continue
        #dst = process_image_lane_detect(image)
        dst = image
        draw_image = np.copy(image)
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32)/255
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_single1_of_2,
                               xy_window=(80, 80), xy_overlap=(0.85, 0.85))
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_single2_of_2,
                               xy_window=(96, 96), xy_overlap=(0.8, 0.8))
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(96, 96), xy_overlap=(0.85, 0.85))
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        # #ystart = y_start_stop[0]
        ystart = 400
        # #ystop = y_start_stop[1]
        ystop = 656
        #scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        scales = [1.5]
        #scale = 1.5
        windows = []
        # for scale in scales:
        #     windows += find_cars2(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        # scale = 1.1
        # hot_windows+= find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        # scale = 1.9
        # hot_windows+= find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        # scale = 0.4
        # hot_windows+= find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=1)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = add_heat(heat, hot_windows)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 3)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        print(labels[1], 'cars found')
        mpimg.imsave(outimgpath + "heatmap_" + os.path.splitext(os.path.basename(filename))[0] + ".png", labels[0])  # jpg write not possible, use png

        draw_img = draw_labeled_bboxes(window_img, labels)
        mpimg.imsave(outimgpath + "candidate_" + os.path.splitext(os.path.basename(filename))[0] + ".png", draw_img)  # jpg write not possible, use png
#        mpimg.imsave('candidates.png', window_img)
        print('processed example img ', filename)

else:
    video_input00 = 'test_video.mp4'
    video_input01 = 'project_video.mp4'
    video_output00 = 'output_videos/test_video_output.mp4'
    video_output01 = 'output_videos/project_video_output.mp4'
    videoclip00 = VideoFileClip(video_input00)
    videoclip01 = VideoFileClip(video_input01)#.subclip(10,10.1)
    processed_video = videoclip00.fl_image(process_image_vedet)
    processed_video = videoclip01.fl_image(process_image_vedet)
    processed_video.write_videofile(video_output00, audio=False)
    processed_video.write_videofile(video_output01, audio=False)




