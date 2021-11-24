import os      
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, VGG19, ResNet152V2, NASNetLarge
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mat
import argparse
import Augmentor
import cv2
import os
import pandas as pd
import shutil
import random
from sklearn.cluster import KMeans
from sklearn import metrics 
from scipy.spatial.distance import cdist
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
import pydicom
from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference
from PIL import Image
import torch
import time

# 切肺的model
model = create_model("resnet34")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

data_path = 'VALID/'
save_path1 = 'VALID1/'
save_path2 = 'VALID2/'
datalist = os.listdir(data_path)

for file_name in datalist:
    print(file_name)
    image, mask = inference.inference(model,data_path + file_name, 0.2)
    temp_img = inference.img_with_masks( image, [mask[0], mask[1]], alpha = 0.1)
    plt.imsave(save_path1 + file_name, temp_img)

    #time.sleep(1)
    image1 = image.copy()
    image2 = image.copy()
    mask[0] = np.where(mask[0]>0, 1, mask[0])
    mask[1] = np.where(mask[1]>0, 1, mask[1])
    image2[:,:,0] = image1[:,:,0]*mask[0] + image1[:,:,0]*mask[1]
    image2[:,:,1] = image1[:,:,1]*mask[0] + image1[:,:,1]*mask[1]
    image2[:,:,2] = image1[:,:,2]*mask[0] + image1[:,:,2]*mask[1]
    print(save_path2 + file_name)
    try:
        plt.imsave(save_path2 + file_name, image2)
    except:
        plt.imsave(save_path2 + file_name, normalize(image2))
    #time.sleep(1)