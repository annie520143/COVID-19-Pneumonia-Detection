# COVID-19-Pneumonia-Detection

### Introduction
Our tsk is to classify chest x-ray image into three class: Normal, Pneumonia, COVID. Our method can devided into four steps:
1. X-ray image enhancement inverse
2. X-ray image enhancement
3. Do lung segmataion
4. Train pretrained model

### File Description
* `lung_segmentation.py`: input original x-ray image and output two type of image, the one is coloring the lung part of the image, another will only reserve the lung part
* `transfer_learning.py`: all pretrained models in https://keras.io/api/applications/?fbclid=IwAR0Cxq_0f_cG01QhSt5CivNj8mCzL65OzmGo7-Rme29EBI0MDkQ2QP5t2CE
* `/x-ray-images-enhancement-master/`
  * `app.py`: run high-frequency emphasis filtering (hef) to enhance the original image

### Requirements
see requirements.txt

### Implementation
1. Inverse image to black lung and white bone (change row 361 : valid_data / covid_data / pneumonia_data / normal_data)
 ```
 python invert.py
 ```
2. X-ray image enhancement
 ```
 python app.py -a hef
 ```
3. Do lung segmataion
 change data in file to the location of image folder that you want to preprocess
 ```
 python lung_segmentation.py
 ```
4. Train pretrained model
 You can change any pre-trained model in link below
 Our best model: DenseNet201
 All models: https://keras.io/api/applications/?fbclid=IwAR0Cxq_0f_cG01QhSt5CivNj8mCzL65OzmGo7-Rme29EBI0MDkQ2QP5t2CE
 ```
 python transfer_learing.py
 ```

### Result
| | w/o Segmentation | w/ Segmentation|
| -------- | -------- | --------  |
| w/o Enhancement | 0.55 | x |
| w/ Enhancement | x | 0.6 |

| | w/o Segmentation | w/ Segmentation|
| -------- | -------- | --------  |
| w/o Enhancement | 0.48 | x |
| w/ Enhancement | x | 0.51 |

 
### Reference
* https://keras.io/api/applications/
* https://www.nature.com/articles/s41598-021-95561-y
* https://github.com/asalmada/x-ray-images-enhancement
* https://github.com/alimbekovKZ/lungs_segmentation_train
