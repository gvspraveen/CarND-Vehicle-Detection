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
[hog9]: ./output_images/hog_9.jpg
[hog11]: ./output_images/hog_11.jpg
[scales_test]: ./output_images/scales_test.jpg
[frames_heatmap]: ./output_images/frames_heatmap.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
Code For Project is mostly in thes files

- [Solution.ipynb](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/solution.ipynb). This contains most of the training and processing pipelines
- [transforms.py](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/transforms.py). This file contains the code for transformations and feature extractions like HOG, spatial, histogram. Some of the
code is from lessons. I added comments in the code to indicate reference to external sources (like lesson code snippets)
- [im_utils.py](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/im_utils.py). This is used to centralize image reading logic.

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Actual code to extract hog features in method `get_hog_features` in file transforms.py](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/transforms.py).

The code looks very similar to one taught in class (with only difference being split into multiple helper methods). 


#### 2. Explain how you settled on your final choice of HOG parameters.

I first started by exploring car and non car test images. This can be found in **Step1 - Data Exploration** in [Solution.ipynb](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/solution.ipynb) notebook..
 
Exploration and trials for different HOG parameters can be found in **Step 2.1 - Visualize Hog transformation** in [Solution.ipynb](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/solution.ipynb) notebook.

 
- Here are the results of Hog transform on few training (car) images. Each column is hog transform in one of the three channel

![Hog 9][hog9]

- Here are the results of Hog transform on few training (car) images with orientation of 11. Each column is hog transform in one of the three channel

![Hog 11][hog11]


In order to help exploring various transformations, i defined the method `extract_features` in [transforms.py](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/transforms.py).
This method is slight extension from code in lesson. It has various parameters to enable/disable spatial, hog, hist transforms. Also configuration for each
of the transforms. It combines all these features and provides final feature vector.

The best combination of the features is determined by using training classifier and comparing speed and accuracy of various combinations. This is explained in next section.

I tried different combinations of feature extractions. The code for this is in section **Step 2.3 - Test various parameter configurations**.
In this section I defined a helper method `test_parameters(option_dict)`. In next set of cells, you will notice how I tried this against
different possible combinations and checking accuracy. 

You will notice that I tuned various knobs

```
HOG orientation = 9, 11
Spatial transform = Enabled, Disabled
Spatial size = (16, 16), (32, 32)
Hist transform = Enabled, Disabled
Color Spaces = RGB, YUV, YCrCb

```
 
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I defined a method `training_pipeline` in **Step 2.2 - Define a Tranining Pipeline**.

The method takes various combinations of transformation parameters. It applied **Linear SVM** on them. I trained this on
car and non car features. Using sklearn's `train_test_split`, I split training and testing group and measured accuracy. This method
returns a tuple `svc, training_accuracy, training_time, X_scaler`. These are used later in the project.

Finally after playing around with various parameter combinations in section **Step 2.3 - Test various parameter configurations**, I finally decided to these parameters.

```
final_cspace='YUV'
final_spatial_size=(16, 16)
final_spatial_transform= True
final_hist_transform= True
final_hist_bins= 32
final_hog_transform= True
final_orient= 11
final_pix_per_cell= 8
final_cell_per_block= 2
final_hog_channel= 'ALL'
```

This model yelded accuracy of 99.2%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For this I defined a helper method `find_cars` in [transforms.py](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/transforms.py).

This method is based of lesson code. It takes ystart, ystop and scaling factor along with all other feature combinations explained in previous sections.

As explained in lecture videos, instead of taking hog transform on every window, in this method we take HOG transform for whole image at once.
Then using subsampling technique we extract HOG features for each sub window.
I then run the trainined classfier for each window and return list of bounding boxes for positive car predictions.


I wrote another wrapper method `run_sliding_window` which takes list of scales and parameters needed for `find_cars`. Internally this method just
calls `find_cars` for each scale. It returns a concatenated list of bounding boxes for each scale.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To test this I used **Step 3.2 - Run Sliding search on test images** in [Solution.ipynb](https://github.com/gvspraveen/CarND-Vehicle-Detection/blob/master/solution.ipynb).

Here I tried various thresholds and scales and printed out bounding boxes and detected cars. The list of scales that worked best for me are

```
Each tuple in this list indicate (ystart, ystop, scale_factor)
[(360, 520, 1.35), (380, 550, 1.5), 
(400, 600, 1.75), (400, 600, 1.85), 
(450, 680, 2.25), (450, 680, 2.15)]

```
Intuition is, cars in top half of image are far away from view and are usually smaller in size. Cars in middle of frame are slightly bigger. And cars in lower frame are bigger
from field of view. However on each of these windows use different scales to avoid misses.

Check following image which shows cars detected on 5 random test images. There are odd false positives. But trying to eliminate
them purely using scales was tough. Rather I had to rely on pipeline to keep track of previous frame results and remove detected anamolies
in any individual frame.

![Cars detected][scales_test]



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For pipeline, I defined a class - `Pipeline`. This class processes each frame. Internally it keeps track of heatmaps in previous frames (running average of past n frames).

In order to even out false positives and wobbly boxes, I explored multiple options.

1. Keep track of bounding boxes from last `n` frames. When applying heatmap for any frame using bounding boxes from previous `n` frames. 
Finally define a threshold which makes sure boxes which appear 70% of `n` frames are considered.

2. Follow suggestion in [discussion board](https://discussions.udacity.com/t/wobbly-box-during-video-detection/231487/3). This basically
scales our heatmap for given frame with heatmap uptil now. Also it adjusted current frame of image to average out with last n frames and then
draws bounding boxes. **But this did not yield good results for me**

3. Keep track of heatmaps from previous n frames. Take a average of heatmaps and then apply threshold.

While #2 did not work well for me. #1 and #3 showed some good results and some issues.

Specifically with #1, I had issue where sometimes fals positives carried over for too long. This was because I was piling on to bounding boxes. So
false negatives also piled up. 

Using #3 Solved this issue. Even with #3 there were few false negatives but they were transient (disappear immediately).

I have tuned and played around with these a lot. But in the end this is best result I could get. May be I will get to it in future.

Here is example of three successive frames with their heatmap in pipelines (using approach just described)

![Frames and Heatmap][frames_heatmap]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem for me was balancing and tuning my parameters between missing real cars vs false positive. If I over correct in favor
of detecting 100% of time (every frame), then I run into problem of too many false positives. Even with smoothening and false positives filtering
techniques I discussed earlier, this was still a issue. On the other hand, if I am too sensitive to false positives, then I run into risk
of having few seconds in video with no boxes on real cars. In the end, I went in favor of detecting cars most of the time and making sure false positives are not too many. 

This is a tradeoff. You might notice in video, when white car emerges from behind (or exiting the frame), boxes are not drawn immediately. They are drawn only when 3/4th of car gets into visible spectrum. This I feel is a fair trade off. May be I will comeback to address this later. But for now I think the solution is reliable.

Another possible approach is liminting the width of search space and eliminating cars on other side of the road.
