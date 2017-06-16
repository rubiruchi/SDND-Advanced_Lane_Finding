## **README**

### **Advanced Lane Finding - An Exercise in Computer Vision**

#### **Victor Roy**

[GitHub Link](https://github.com/soniccrhyme/SDND-Project_4)

---


[//]: # (Image References)

[image1]: ./report_images/camera_calibration.png "Camera Calibration"
[image2]: ./report_images/perspective_transform.png "Perspective Transform"
[sobels]: ./report_images/sobels.png "Sobel Gradients"
[sobel_combined]: ./report_images/sobel_combined.png "Sobel Combined"
[rgbs]: ./report_images/rgbs.png "RGB Binary"
[hlss]: ./report_images/hlss.png "HLS Binary"
[hsvs]: ./report_images/hsvs.png "HSV Binary"
[sv]: ./report_images/hlsandv.png "S & V Combined"
[avg_binary]: ./reort_images/avg_and_binary.png "Averaged & Binary"
[final_binary]: ./report_images/final_binary.png "Final Binary"
[output]: /report_images/output_eg.png "Output Example"
---
### Brief Project Description

This code detects lane markings, approximates their direction and curvature with a second-degree polynomial and overlays those resultant lane lines atop the video stream taken from a center-mounted dashboard camera.   

Frames from the video stream are undistorted via camera calibration. Various color encodings, channels and gradient calculations and are combined in order to create a binary image which extracts lane markings. The image is perspective transformed into a top-down view of the road, wherein the lane markers should appear parallel. The binary image is then fed into a lane detection algorithm which uses nonzero pixel densities to trace the lane's direction and curvature. A second-degree polynomial is fit to both the left and right hand-side lanes. The resultant lane lines (along with text describing lane radii and car offset from center) are then overlaid upon the undistorted video feed and output to an .mp4.

All code references point to Lane_Finder.ipynb. The Lane() class can be found in line.py.

---

### Pipeline

#### 1. Camera Calibration

The project folder (see Udacity's github link above) provided camera calibration images of a chessboard taken at various angles and distances by the camera used for recording the dashboard footage. These chessboard images are fed into OpenCV's camera calibration module. In essence, chessboard calibration uses the fact that each chessboard square is in fact a square, that adjacent rows of checkers are parallel, and that the chessboard in each of the images is the same. The algorithm finds corners of the chessboard, and uses the above assumptions to calculate the matrix coefficient which would undistort the camera image. The most noticeable effects of this distortion appear towards the edge of an image, as noted in the figure below:

![Camera Calibration][image1]

Code for this process found in Input[2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Frames are perspective transformed in order to provide a top-down view of the road section in question. Finding the transformation matrix begins with selecting source points of a trapezoidal polygon on the original image which you would want to be looking down upon in the transformed image; these points were manually selected with the logic of including at least 2 dashed lane markers. The destination points were chosen to allow sufficient width between the lanes in the top-down view. The process is depicted in the figure below; the code for the perspective transform can be found in Input[7]

![Perspective Transform][image2]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In order to extract lane markings from the camera feed, the code uses a mixture of Sobel gradients (in x, y as well as in magnitude and direction) in addition to three different color encodings - RGB, HSV, & HSL. In order to aggregate information provided by the various channels, I used thresholding to create binary images of different color channels; those binary images were conjoined through binary logic operations and averaging.

Images showing the various thresholded binary images of different color channels and gradients are shown below. The exact logic for combining these various factors into a final binary image for lane detection is as follows (excerpted from Input[8]):

```python
sobel_combined[((sobelx == 1) & (dirt == 1)) & (mag == 1)] = 1

RGB_combined[(R_binary == 1) & (G_binary == 1) | (B_binary ==1)] = 1

H, L, S = get_HLS_channels(img)
S_binary = get_binary(S, thresh = S_thresh)

H2, S2, V = get_HSV_channels(img)
SV_binary = np.zeros_like(sobelx)
SV_binary[(S_binary == 1) & (V_binary == 1)] = 1

avg_img = img_as_ubyte(np.average([S_binary, V_binary, RGB_combined, S_binary_2, combined], axis = 0))
avg_binary = get_binary(avg_img, thresh = avg_thresh)

final_avg = img_as_ubyte(np.average([SV_binary, avg_binary], axis = 0))
final_binary = get_binary(final_avg, thresh = final_thresh)
```
![Sobel Gradients, Binary][sobels]
![Sobel Combined][sobel_combined]
![RGB Channels, Binary][rgbs]
![HLS Channels, Binary][hlss]
![HSV Channels, Binary][hsvs]
![SV Combined, Binary][sv]
![Average of Many][avg_binary]

And the final binary image, used for lane detection:

![Final Binary][final_binary]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For the first frame, I conducted a window search which identified high density regions of nonzero pixel values. Once the first frame is processed and its lane markings identified, subsequent frames are processed by simply searching in a margin around the (average of) previous lane markings. Information for each lane is saved in a(n unfortunately) global variable such that it can be recalled throughout the video processing method. These lane finding methods can be found in Input[10] (first frame) and Input[11] (all other frames)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature for the second degree polynomial estimating the lane marking was calculated using the method described [here][http://www.intmath.com/applications-differentiation/8-radius-curvature.php]. Units were converted from pixels to meters using two pieces of important information: 1) the [standard lane width on US highways is 12 feet or 3.657 meters][https://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm] and 2) the [length of a dashed lane marking is 10 feet or 3.04 meters][http://www.ctre.iastate.edu/pubs/itcd/pavement%20markings.pdf]. From these two pieces of information, it is elementary to derive how many vertical and horizontal pixels represent a meter. The code for calculating lane curvature can be found in Input[9]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![Output Example][output]

### Video

[Video Link][./result.mp4]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
