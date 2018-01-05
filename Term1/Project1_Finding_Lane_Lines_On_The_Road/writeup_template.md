# **Finding Lane Lines on the Road** 

## Writeup Template


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:<br>
 * Make a pipeline that finds lane lines on the road<br>
 * Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg "Grayscale"
[image2]: ./test_images_output/solidWhiteRight.jpg "Grayscale"
[image3]: ./test_images_output/solidYellowCurve.jpg "Grayscale"
[image4]: ./test_images_output/solidYellowCurve2.jpg "Grayscale"
[image5]: ./test_images_output/solidYellowLeft.jpg "Grayscale"
[image6]: ./test_images_output/whiteCarLaneSwitch.jpg "Grayscale"

---

### Reflection

### 1. Description of the pipeline including the modification of the draw_lines() function.

My pipeline consisted of 7 steps. 

1. Isolating interesting colors (bright enough of close to yellow for yellow lanes)
2. Convert to grayscale for further processing
3. Blur the image with a Gaussian filter
4. Canny Edge detection
5. Mask intermediate result with a trapezoidal shape to eliminate false detections outside the lane
6. Perform Hough transform
7. Combined rendering of original image with result of hough transform

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:

1. Dividing line segments into left and right half in dependence of the slope
2. Calculated positions of the line segments by a simple midpoint rule
3. Calculated linear regression of these midpoints (or: polynomial fit of degree 1)
4. Draw this linear regression line into the image, extrapolated to appropriate start/end position

Here are some of the demo images to show how the pipeline works:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image5]
![alt text][image6]

Remark: It can be switched back to the original hough transform without the extrapolated one line by calling "hough_lines" instead of "hough_lines_improved".

### 2. Identification of potential shortcomings with the current pipeline

One potential shortcoming would be what would happen when e.g. lighting conditions change. The pipeline currently relies on some parameters which might work for these sequences but for others not.

Another shortcoming could be that this extrapolating to a linear regression only works for non-curved lane lines.

Also, the slope of the detected lanes seems to "flicker". One could reduce this by further parameter optimization but it is not guaranteed that it works for another sequence. Better would be to (Kalman)-filter the current result with the previous ones.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to extend the lane detection to curves. Currently it is restricted to straight driving.

Another potential improvement could be to fine-tune the parameters of canny, hough to be independent of a color segmentation. Also, one could go to different color spaces such as HSI or YUV. RGB is a color space where brightness is not on one of the principal axes. It would be better to color segment based on brightness/luminance and not on specific colors such as yellow. This is pretty country-specific, in other countries, there are other colors.
