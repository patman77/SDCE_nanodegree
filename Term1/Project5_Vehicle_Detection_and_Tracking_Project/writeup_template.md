## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[reportimage01]: ./report_images/figure01.png
[reportimage02]: ./report_images/figure02.png

[test01]: ./test_images/test1.jpg
[reportcandtest01]: ./report_images/candidate_test1.png
[reportheattest01]: ./report_images/heatmap_test1.png
[test02]: ./test_images/test2.jpg
[reportcandtest02]: ./report_images/candidate_test2.png
[reportheattest02]: ./report_images/heatmap_test2.png
[test03]: ./test_images/test3.jpg
[reportcandtest03]: ./report_images/candidate_test3.png
[reportheattest03]: ./report_images/heatmap_test3.png
[test04]: ./test_images/test4.jpg
[reportcandtest04]: ./report_images/candidate_test4.png
[reportheattest04]: ./report_images/heatmap_test4.png
[test05]: ./test_images/test5.jpg
[reportcandtest05]: ./report_images/candidate_test5.png
[reportheattest05]: ./report_images/heatmap_test5.png
[test06]: ./test_images/test6.jpg
[reportcandtest06]: ./report_images/candidate_test6.png
[reportheattest06]: ./report_images/heatmap_test6.png



[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Overall, as for the previous project, I extracted all of the source code from the lessons for later usage. It's contained in the files
"HOG_classify", "car_notcar.py", "color_classify.py", "color_histogram.py", "draw_boxes.py", "get_hog.py", "heatmap.py",
"hog_subsample.py", "lesson_functions.py", "lesson_functions2.py", "norm_shuffle.py", "search_classify.py",
"sliding_window.py", "spatial_bin.py", "template_match.py", "template_matching.py", and finally "vedet_main.py".

 
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file "vedet_main.py", around lines 173. I extract the HOG features for the car and the non-car images.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of each of the `vehicle` and `non-vehicle` classes:



![alt text][reportimage01]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` (only channel 0):

![alt text][reportimage02]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters manually and followed the advice by my last reviewer to choose the color space YCrCb.
Finally, I got a test accuracy of 99.09 % with the following parameters:

```
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
```

See also README.txt for some other combinations.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the source code from the lecture. I took this from the lecture source code search_classify.py.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I took the functions slide_window" and "search_window" from the lecture code.

The overlap I took from the lecture as 0.5, in the beginning. But then I noticed that I missed a lot of potental car detections. So I increased to 0.8 which gave much better results.
For the scaling, it's still on one scaling. I added and used the HOG_subsample.py from the lecture, but didn't find the optimal parameters yet.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

See section below with the heatmaps.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
For the  [first video](./test_video.mp4), here's a [link to my video result](./output_videos/test_video_output.mp4).
For the  [second video](./project_video.mp4), here's a [link to my video result](./output_videos/project_video_output.mp4).

I also uploaded the result videos to Youtube:
- [Youtube video 1](https://youtu.be/llx6qXGSfsM) 
- [Youtube video 2](https://youtu.be/qfEL9dRDo-0) 

In the second larger video there are a lot of issues, still. I think I have to extend to search on different scales.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding detections as bounding boxes and the heatmaps (even the cars on the other side are detected):

![alt_text][test01]
![alt_text][reportcandtest01]
![alt_text][reportheattest01]
![alt_text][test02]
![alt_text][reportcandtest02]
![alt_text][reportheattest02]
![alt_text][test03]
![alt_text][reportcandtest03]
![alt_text][reportheattest03]
![alt_text][test04]
![alt_text][reportcandtest04]
![alt_text][reportheattest04]
![alt_text][test05]
![alt_text][reportcandtest05]
![alt_text][reportheattest05]
![alt_text][test06]
![alt_text][reportcandtest06]
![alt_text][reportheattest06]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further:

I think the pipeline roughly works to detect vehicles. There are some false positives for cars on the other side.
This could be restricted by the result from the previous project "Lane Finding".

Also, there were some (seldom) false positives on the street, which could be avoided by "negative mining": take this regions from the video and add it to the noncar images.

