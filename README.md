# COVID-19-Pneumonia-Detection

### Introduction

### File Description
* `lung_segmentation.py`: input original x-ray image and output two type of image, the one is coloring the lung part of the image, another will only reserve the lung part
* `transfer_learning.py`: pretrained model for KGG16, KGG19, ...
* `/x-ray-images-enhancement-master/`
  * `app.py`: run high-frequency emphasis filtering (hef) to enhance the original image

### Requirements
see requirements.txt

### Implementation
1. Inverse image to black lung and white bone
 ```
 
 ```
2. X-ray image enhancement
3. Do lung segmataion
 change data in file to the location of image folder that you want to preprocess
 ```
 python lung_segmentation.py
 ```
4. Run pretrained model
 ```
 
 ```
