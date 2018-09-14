import matplotlib.image as mpimg

from apply_sobel import abs_sobel_thresh
from color_and_gradient import pipeline
from curve_pixels import measure_curvature_pixels
from direction_gradient import dir_threshold
from lane_histogram import hist


image = mpimg.imread('test_images/test1.jpg')
