------------------------------------------------------------------
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 64  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 707]  # Min and max in y to search in slide_window()

Test Accuracy of SVC =  0.9801
------------------------------------------------------------------


------------------------------------------------------------------
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 64  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 707]  # Min and max in y to search in slide_window()


Using: 10 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6840
6.71 Seconds to train SVC...
Test Accuracy of SVC =  0.9909
My SVC predicts:  [ 1.  1.  0.  1.  1.  1.  1.  0.  1.  0.]
For these 10 labels:  [ 1.  1.  0.  1.  1.  1.  1.  0.  1.  0.]
0.06847 Seconds to predict 10 labels with SVC
------------------------------------------------------------------
