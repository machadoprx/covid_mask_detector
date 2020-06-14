# covid_mask_detector
This project aims to, both, detect masks in a human face using a CNN, in a image of a public area, that may serve as sensor for the Corona Virus state in that region, and avaliating the impact of the use of image processing techniques in the data set of this problem, such as accuracy difference, cost/benefit ratio.

The assets used in this code includes 2 CNNs, one for mask detection coded and trained by me in the project: https://github.com/machadoprx/CoolConvNN

One for face detection:
https://github.com/ipazc/mtcnn

The face detection can also be performed by the Viola Jones algorithm instead, besides its lower accuracy the VJ is faster.

# CNN Architecture for mask detection

<p align="center">
	<img src="nn.png" width="100%" heigth="100%" alt="cnnarc"></img>
</p>

# Input and output example:

Detected faces will be evaluated , the boxes are green for people with masks and red for unmasked

<div class="row" align="center">
  <div class="column" align="center">
    <img src="das.jpg" alt="in" width="40%" heigth="40%">
  </div>
  <div class="column"align="center">
    <img src="out.png" alt="out" width="40%" heigth="40%">
  </div>
</div>

# How to use:

Clone the repository

```
$ git clone https://github.com/machadoprx/covid_mask_detector.git
```

Build new image data set for training in CoolConvNet (https://github.com/machadoprx/CoolConvNN)

```
$ python3 data_set_enhancer.py make_data_set filter_name src_path
```

The available filters currently are laplace sharpening and clahe(https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)

For this application the CNN is already trained for mask detection, for further information about the framework and its use check the CoolConvNet repo.

Evaluate image, detecting faces and drawing boxes (red for unmasked and green for masked)

```
$ python3 data_set_enhancer.py test_image filter_name src_path
```



