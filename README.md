## **README**

### **Advanced Lane Finding - An Exercise in Computer Vision**

#### **Victor Roy**

[GitHub Link](https://github.com/soniccrhyme/SDND-Project_4)

[//]: # (Image References)

[image1]: ./report_images/camera_calibration.png "Camera Calibration"
[image2]: ./report_images/perspective_transform.png "Perspective Transform"
[sobels]: ./report_images/sobels.png "Sobel Gradients"
[sobel_combined]: ./report_images/sobel_combined.png "Sobel Combined"
[rgbs]: ./report_images/rgbs.png "RGB Binary"
[rgb_combined]: ./report_images/rgb_combined.png
[hlss]: ./report_images/hlss.png "HLS Binary"
[hsvs]: ./report_images/hsvs.png "HSV Binary"
[sv]: ./report_images/hlsandv.png "S & V Combined"
[avg_binary]: ./report_images/avg_and_binary.png "Averaged & Binary"
[final_binary]: ./report_images/final_binary.png "Final Binary"
[output]: /report_images/output_eg.png "Output Example"

---
### Brief Project Description

This code detects lane markings, approximates their direction and curvature with a second-degree polynomial and overlays those resultant lane lines atop the video stream taken from a center-mounted dashboard camera.   

Frames from the video stream are undistorted via camera calibration. Various color encodings, channels and gradient calculations and are combined in order to create a binary image which extracts lane markings. The image is perspective transformed into a top-down view of the road, wherein the lane markers should appear parallel. The binary image is then fed into a lane detection algorithm which uses nonzero pixel densities to trace the lane's direction and curvature. A second-degree polynomial is fit to both the left and right hand-side lanes. The resultant lane lines (along with text describing lane radii and car offset from center) are then overlaid upon the undistorted video feed and output to an .mp4.

All code references point to Lane_Finder.ipynb. The Lane() class can be found in line.py.

---

### Pipeline

Code for the pipeline is found in Input[14]. The pipeline itself calls several methods, from camera calibration to calculating the radius of curvature. These pipeline methods are described below.

#### 1. Camera Calibration

The project folder (see Udacity's github link above) provided camera calibration images of a chessboard taken at various angles and distances by the camera used for recording the dashboard footage. These chessboard images are fed into OpenCV's camera calibration module. In essence, chessboard calibration uses the fact that each chessboard square is in fact a square, that adjacent rows of checkers are parallel, and that the chessboard in each of the images is the same. The algorithm finds corners of the chessboard, and uses the above assumptions to calculate the matrix coefficient which would undistort the camera image. The most noticeable effects of this distortion appear towards the edge of an image, as noted in the figure below:

![Camera Calibration][image1]

Code for this process found in Input[2]; camera calibration is also completed before entering the video processing pipeline, in Input[13]

#### 2. Perspective Transform

Frames are perspective transformed in order to provide a top-down view of the road section in question. Finding the transformation matrix begins with selecting source points of a trapezoidal polygon on the original image which you would want to be looking down upon in the transformed image; these points were manually selected with the logic of including at least 2 dashed lane markers. The destination points were chosen to allow sufficient width between the lanes in the top-down view. The process is depicted in the figure below; the code for the perspective transform can be found in Input[7]

![Perspective Transform][image2]

#### 3. Creating a Thresholded Binary Image

In order to extract lane markings from the camera feed, the code uses a mixture of Sobel gradients (in x, y as well as in magnitude and direction) in addition to three different color encodings - RGB, HSV, & HSL. In order to aggregate information provided by the various channels, I used thresholding to create binary images of different color channels; those binary images were conjoined through binary logic operations and averaging.

Images showing the various thresholded binary images of different color channels and gradients are shown below. The exact logic for combining these various factors into a final binary image for lane detection is as follows (excerpted from Input[8]):

```python
sobel_combined[((sobelx == 1) & (dirt == 1)) & (mag == 1)] = 1

RGB_combined[(R_binary == 1) & (G_binary == 1) | (B_binary ==1)] = 1

H, L, S = get_HLS_channels(img)
S_binary = get_binary(S, thresh = S_thresh)

H2, S2, V = get_HSV_channels(img)

SV_binary[(S_binary == 1) & (V_binary == 1)] = 1

avg_img = img_as_ubyte(np.average([S_binary, V_binary, RGB_combined, S_binary_2, sobel_combined], axis = 0))
avg_binary = get_binary(avg_img, thresh = avg_thresh)

final_avg = img_as_ubyte(np.average([SV_binary, avg_binary], axis = 0))
final_binary = get_binary(final_avg, thresh = final_thresh)
```
Sobel Gradients:  
![Sobel Gradients, Binary][sobels]
Combining Sobel X with Gradient Magnitude & Direction:  
![Sobel Combined][sobel_combined]
RGB Channels:  
![RGB Channels, Binary][rgbs]
RGB, Combined:  
![RGB, Combined][rgb_combined]
HLS Channls:  
![HLS Channels, Binary][hlss]
HSV Channels:  
![HSV Channels, Binary][hsvs]
(HL)S & V Channel, combined:  
![SV Combined, Binary][sv]
Averaging several of the binary images:  
![Average of Many][avg_binary]

And the final binary image, used for lane detection:

![Final Binary][final_binary]

#### 4. Lane Marking Detection and Lane Line Fitting

For the first frame, I conducted a window search which identified high density regions of nonzero pixel values. Once the first frame is processed and its lane markings identified, subsequent frames are processed by simply searching in a margin around the (average of) previous lane markings. Information for each lane is saved in a(n unfortunately) global variable such that it can be recalled throughout the video processing method. These lane finding methods can be found in Input[10] (first frame) and Input[11] (all other frames)

#### 5. Radius of Curvature

Radius of curvature for the second degree polynomial estimating the lane marking was calculated using the method described [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). Units were converted from pixels to meters using two pieces of important information: 1) the [standard lane width on US highways is 12 feet or 3.657 meters](https://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm) and 2) the [length of a dashed lane marking is 10 feet or 3.04 meters](http://www.ctre.iastate.edu/pubs/itcd/pavement%20markings.pdf). From these two pieces of information, it is elementary to derive how many vertical and horizontal pixels represent a meter. The code for calculating lane curvature can be found in Input[9]

#### 6. Output screenshot

![Output Example][output]

### Video

Project Video: ([GitHub](https://github.com/soniccrhyme/SDND-Project_4/blob/master/result.mp4)/[YouTube](https://youtu.be/2wW73px649M))  
Challenge Video: ([GitHub](https://github.com/soniccrhyme/SDND-Project_4/blob/master/result_challenge.mp4)/[YouTube](https://youtu.be/8aydAjBwqHE))

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  

The first major challenge was trying to figure out the best way to combine the different sobel gradients and color channels to robustly extract the lane markings. Each channel or gradient had its own unique piece of information; some were better at dealing with white lines, others, yellow; some worked well when contrast was high, others did not. A lot of trial and error led to the solution I have now.

And it worked almost flawlessly on the project video _except_ when transitioning from concrete to tarmac. In some of these frames, there was just too much noise or not enough consistent contrast to be able to pick up the lane markings effectively. To solve this I added memory by instantiating a global lane object - one for the right-hand side lane, another for the left-hand side lane. To smooth out lane tracking, I not only used an _average of a weighted average_ across the past 11 frames, but also threw out lines where the curvature was drastically different than that running average.

These methods combined allow the code as written to do well on both the project video and the challenge video. When watching the challenge video result, note how in the first few frames, the ends of the lane are a bit too curved, but also how over the next several frames that artifact is eventually smoothed away.  

The code as written, however, still fails rather miserably on the harder challenge video. Part of the problem here is that the perspective transform looks _too_ far ahead, and so when the road curves more dramatically, the mask is picking up pieces of the other lane, the shoulder, or, sometimes, mostly just ditch foliage and the trees beyond. It would be desirable to somehow implement a dynamic forward sight range depending on what the car knows about the general tortuousness of the road it is on. It would also be prudent to add additional safeguards against detecting more wild changes in road curvature, or at least making sure the right-hand side lane is always on the right of the left-hand side lane, and that the two never aught cross.

Other potential problems which became apparent when attempting to process the harder challenge video was how the code dealt with low contrast situations (such as when there is much glare from the sun, or when the car is in shadows, or when the horizon has both extreme brightness and shadow). Analogous to having dynamic forward sight range for various levels of road twistiness, it would be nice to implement dynamic image processing for different lighting conditions, such as using color different channels or different thresholds.
