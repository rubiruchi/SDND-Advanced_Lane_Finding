## **README**

### **Advanced Lane Finding - An Exercise in Computer Vision**

#### **Victor Roy**

[GitHub Link](https://github.com/soniccrhyme/SDND-Project_4)

---


[//]: # (Image References)

[image1]: ./graphics/train-count_boxplot.png "Count in Train Boxplot"
[image2]: ./graphics/rand-images.png "Sample of Images"
[image3]: ./graphics/model_architecture.png "Model Architecture"
[image4]: ./graphics/custom_roadsigns.png "German Traffic Signs taken from Google Streetview"
[image5]: ./graphics/custom_roadsigns_predictions.png "Predictions for Traffic Signs taken from Google Streetview"
[image6]: ./german_roadsigns/test_sign_3.png "Traffic Sign 3"
[image7]: ./german_roadsigns/test_sign_4.png "Traffic Sign 4"
[image8]: ./german_roadsigns/test_sign_5.png "Traffic Sign 5"

---
### Brief Project Description

This code detects lane markings and overlays them atop the video stream of a center-mounted dashboard camera.

Frames from the video stream are undistorted via camera calibration. Various color encodings, channels and gradient calculations and are combined in order to create a binary image which captures lane markings. The image is perspective transformed into a top-down view of the road, wherein the lane markers should appear parallel. The binary image is then fed into a lane detection algorithm which uses nonzero pixel densities to find lane markings. A second-degree polynomial is fit to both the left and right hand-side lanes. The resultant lane lines (along with text describing lane radii and car offset from center) are then overlaid upon the undistorted video feed and output to an .mp4.
---



### Pipeline

#### 1. Camera Calibration

The project folder (see Udacity's github link above) provided camera calibration images of a chessboard taken at various angles and distances by the camera used for recording the dashboard footage. These chessboard images are fed into OpenCV's camera calibration module. In essence, chessboard calibration uses the fact that each chessboard square is, in fact a square, that adjacent rows of checkers are parallel, and that the chessboard being pictured in each of the images is in fact the same. The algorithm finds corners of the chessboard, and uses the above assumptions to calculate the matrix coefficient which would undistort the camera image. The most noticeable effects of this distortion appear towards the edge of an image.

Code for this process found in [REF]

Before/after images of applying undistortion [INSERT]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Note why 2 & 3 are reversed.

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

### Video

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

---

### References
