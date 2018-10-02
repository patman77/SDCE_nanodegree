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
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from apply_sobel import abs_sobel_thresh
from color_and_gradient import pipeline
#from curve_pixels import measure_curvature_pixels
from direction_gradient import dir_threshold
from lane_histogram import hist
from magnitude_gradient import mag_thresh
from prev_poly import fit_poly, search_around_poly
from radius_curve import measure_curvature_real2
from rgb_to_hls import hls_select
from sliding_window import find_lane_pixels, fit_polynomial, fit_polynomial2
from sliding_window_template import window_mask,find_window_centroids
from undistort import cal_undistort
from warp import corners_unwarp_improved
#from findchessboardcorners import nx, ny
nx = 9
ny = 6
calibimgpath = './camera_cal/calibration*.jpg'
calibfilename = 'calib.p'
#some params for plots
horiz_spacing = .2
width_space = .05




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



def backproject_measurement(warped, ploty, left_fitx, right_fitx, Minv, undist):
    """ projects measurements back down onto the road
    :param warped    : warped binary image
    :param ploty     : lane line pixels, y-range
    :param left_fitx : x pixel values of the left fitted line
    :param right_fitx: x pixel values of the right fitted line
    :param Minv      : inverse perspective transform
    :param undist    : original image
    :return:
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    #mpimg.imsave("final_result.png", result)
    return result

def draw_curvature_and_position(img, curvature_radius, center_distance = 42.0):
    copied_img = np.copy(img)
    height = copied_img.shape[0]
    font = cv2.FONT_HERSHEY_COMPLEX
    topLeftCornerOfText  = (20, 40)
    topLeftCornerOfText2 = (20, 70)
    fontScale = 1
    fontColorOutline = (255, 255, 255)
    fontColor = (0, 0, 0)
    lineType = cv2.LINE_8
    lineType2 = cv2.LINE_4
    text = "Curvature Radius = " + '{:05.2f}'.format(curvature_radius) + " m"
    # outline fonts taken from https://stackoverflow.com/questions/48516211/how-to-show-white-text-on-an-image-with-black-border-using-opencv2
    #size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2, 4)[0]
    cv2.putText(copied_img, text, topLeftCornerOfText,
                font, fontScale,
                fontColor, lineType)
    cv2.putText(copied_img, text, topLeftCornerOfText,
                font, fontScale,
                fontColorOutline, lineType2)
    abs_center_distance = abs(center_distance)
    dir = ''
    if center_distance > 0:
        direction = 'right'

    elif center_distance < 0:
        direction = 'left'
    else:
        direction = "perfectly in the middle of "
    text2 = 'Vehicle Position: ' + '{:05.2f}'.format(abs_center_distance) + 'm ' + direction + ' of center'
    # cv2.putText(copied_img, text2, topLeftCornerOfText2,
    #             font, fontScale,
    #             fontColor, lineType)
    # cv2.putText(copied_img, text2, topLeftCornerOfText2,
    #             font, fontScale,
    #             fontColorOutline, lineType2)
    plt.imshow(copied_img)
    #mpimg.imsave("final_result2.png", copied_img)
    return copied_img

def process_image(image):
    """

    :param image: image to be processed, going through the entire pipeline: undistort, color/gradient thresholding, warp, lane detect, curvature calculation
    :return: processed imaged
    """
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    color_binary = pipeline(dst)
    unwarped, M, Minv = corners_unwarp_improved(color_binary, nx, ny, mtx, dist)
    warped_gray = cv2.cvtColor(unwarped, cv2.COLOR_RGB2GRAY)
    out_img, ploty, left_fitx, right_fitx = fit_polynomial2(warped_gray)
    left_curverad, right_curverad = measure_curvature_real2(ploty, left_fitx, right_fitx)
    result = backproject_measurement(warped_gray, ploty, left_fitx, right_fitx, Minv, image)
    final_result = draw_curvature_and_position(result, left_curverad)
    return final_result

# -----------------------------------------------------------------------------
# ----------------------------------start of main -----------------------------
# -----------------------------------------------------------------------------

try:
    if Path(calibfilename).is_file():
        print("Found existing calibration file under the given name", calibfilename, ", using that one")
        dist_pickle = pickle.load(open(calibfilename, "rb"))
        mtx  = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

    else:
        print("No existing calibration file under the given name", calibfilename, "recalculating")
        mtx, dist = calibrate_camera(calibimgpath)
except:
    print('exception!!')
    exit(1)

testsingleimage = False

if testsingleimage == False:
    print('starting video pipeline')
    video_input01 = 'project_video.mp4'
    video_input02 = 'challenge_video.mp4'
    video_input03 = 'harder_challenge_video.mp4'
    video_output01 = 'output_videos/project_video_output.mp4'
    video_output02 = 'output_videos/challenge_video_output.mp4'
    video_output03 = 'output_videos/harder_challenge_video_output.mp4'
    videoclip01 = VideoFileClip(video_input01)
    videoclip02 = VideoFileClip(video_input02)
    videoclip03 = VideoFileClip(video_input03)

    processed_video = videoclip01.fl_image(process_image)
    processed_video.write_videofile(video_output01, audio=False)
    # processed_video = videoclip02.fl_image(process_image)
    # processed_video.write_videofile(video_output02, audio=False)
    # processed_video = videoclip03.fl_image(process_image)
    # processed_video.write_videofile(video_output03, audio=False)

    exit(0)


# test undistortion
# step 2: Apply a distortion correction to raw images.
#testimg = mpimg.imread('./camera_cal/calibration1.jpg')
#testimg = mpimg.imread('./test_images/test2.jpg')
testimg = mpimg.imread('./test_images/straight_lines1.jpg') # for determining the trapezoid for unwarping
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
mpimg.imsave("undistorted.png", dst)
cv2.waitKey(100000)

# step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
color_binary = pipeline(dst)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
f.subplots_adjust(hspace = horiz_spacing, wspace=width_space)
ax1.set_title("original image")
ax1.axis('off')
ax1.imshow(testimg)
ax2.set_title("color/gradient thresholded image")
ax2.axis('off')
ax2.imshow(color_binary)
print(" ")
cv2.waitKey(100000)

# step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
unwarped, M, Minv = corners_unwarp_improved(color_binary, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
f.subplots_adjust(hspace = horiz_spacing, wspace=width_space)
ax1.set_title("original image")
ax1.axis('off')
ax1.imshow(color_binary)
ax2.set_title("warped image")
ax2.axis('off')
ax2.imshow(unwarped)
mpimg.imsave("unwarped.png", unwarped)
print(" ")
cv2.waitKey(100000)

# step 5: Detect lane pixels and fit to find the lane boundary.
from sliding_window_template import window_width, window_height, margin
warped_gray = cv2.cvtColor(unwarped, cv2.COLOR_RGB2GRAY)
out_img, ploty, left_fitx, right_fitx = fit_polynomial2(warped_gray)

plt.imshow(out_img)
mpimg.imsave("test.png", out_img)
print(' ')
# window_centroids = find_window_centroids(warped_gray, window_width, window_height, margin)
# # If we found any window centers
# if len(window_centroids) > 0:
#
#     # Points used to draw all the left and right windows
#     l_points = np.zeros_like(warped_gray)
#     r_points = np.zeros_like(warped_gray)
#
#     # Go through each level and draw the windows
#     for level in range(0, len(window_centroids)):
#         # Window_mask is a function to draw window areas
#         l_mask = window_mask(window_width, window_height, warped_gray, window_centroids[level][0], level)
#         r_mask = window_mask(window_width, window_height, warped_gray, window_centroids[level][1], level)
#         # Add graphic points from window mask here to total pixels found
#         l_points[(l_points == 255) | ((l_mask == 1))] = 255
#         r_points[(r_points == 255) | ((r_mask == 1))] = 255
#
#     # Draw the results
#     template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
#     zero_channel = np.zeros_like(template)  # create a zero color channel
#     template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
#     warpage = np.dstack((warped_gray, warped_gray, warped_gray)) * 255  # making the original road pixels 3 color channels
#     output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results
#
# # If no window centers found, just display orginal road image
# else:
#     output = np.array(cv2.merge((warped_gray, warped_gray, warped_gray)), np.uint8)
#
# # Display the final results
# plt.imshow(output)
# plt.title('window fitting results')
# plt.show()
# print(' ')


# step 6: Determine the/ curvature of the lane and vehicle position with respect to center.
# Calculate the radius of curvature in meters for both lane lines
left_curverad, right_curverad = measure_curvature_real2(ploty, left_fitx, right_fitx)

print(left_curverad, 'm', right_curverad, 'm')
# Should see values of 533.75 and 648.16 here, if using
# the default `generate_data` function with given seed number

# step 7: Warp the detected lane boundaries back onto the original image.
result = backproject_measurement(warped_gray, ploty, left_fitx, right_fitx, Minv, testimg)
final_result = draw_curvature_and_position(result, left_curverad)
print('end')

# step 8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


