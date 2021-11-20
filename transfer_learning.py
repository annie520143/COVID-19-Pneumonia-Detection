
# %%
import os      
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, VGG19, ResNet152V2, NASNetLarge
#from tensorflow.keras.applications import EfficientNetB1
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
#%%

'''
這部分的為圖片前處理，會將 train 的檔案處理完後分成 NORMAL PNEUMONIA COVID 放入對應的資料夾
然後 valid 處理完則是放入 VALID 的資料夾
以 VALID 為例 : VALID 為黑白圖 ，VALID1 為黑白圖搭配標示，VALID2 為單獨只有肺部圖片
'''

data = pd.read_csv('data_info.csv')
trian_data_path = ('data/data/train/')
valid_data_path = ('data/data/valid/')
train_dir = os.listdir(trian_data_path)
valid_dir = os.listdir(valid_data_path)
png_path = ''

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

for dir_name in valid_dir:
    dir2_name = os.listdir(valid_data_path + dir_name + '/')
    file_name = os.listdir(valid_data_path + dir_name + '/' + dir2_name[0] + '/')
    ds = pydicom.read_file(valid_data_path + dir_name + '/' + dir2_name[0] + '/' + file_name[0])
    img = ds.pixel_array
    save_name = 'VALID/' + file_name[0].replace('.dcm', '.png')
    save_name1 = 'VALID1/' + file_name[0].replace('.dcm', '.png')
    save_name2 = 'VALID2/' + file_name[0].replace('.dcm', '.png')
    plt.imsave(save_name, img)
    time.sleep(2)
    img = Image.open(save_name).convert('L')
    img.save(save_name) ############ 1 gray
    time.sleep(2)
    image, mask = inference.inference(model,save_name, 0.2)
    temp_img = inference.img_with_masks( image, [mask[0], mask[1]], alpha = 0.1)
    plt.imsave(save_name1, temp_img) ############ 2 mark
    image1 = image.copy()
    image2 = image.copy()
    mask[0] = np.where(mask[0]>0, 1, mask[0])
    mask[1] = np.where(mask[1]>0, 1, mask[1])
    image2[:,:,0] = image1[:,:,0]*mask[0] + image1[:,:,0]*mask[1]
    image2[:,:,1] = image1[:,:,1]*mask[0] + image1[:,:,1]*mask[1]
    image2[:,:,2] = image1[:,:,2]*mask[0] + image1[:,:,2]*mask[1]
    try:
        plt.imsave(save_name2, image2) ############ 3 lung
    except:
        plt.imsave(save_name2, normalize(image2)) ############ 3 lung
    time.sleep(2)

for dir_name in train_dir:
    dir2_name = os.listdir(trian_data_path + dir_name + '/')
    file_name = os.listdir(trian_data_path + dir_name + '/' + dir2_name[0] + '/')
    search_file = file_name[0].split('.')[0]
    answer_list = (data.loc[data['FileID']==search_file]).values.tolist()[0]
    class_num = answer_list.index(1)
    if class_num == 1:
        png_path = 'NORMAL/'
        png_path1 = 'NORMAL1/'
        png_path2 = 'NORMAL2/'
    elif class_num == 2:
        png_path = 'PNEUMONIA/'
        png_path1 = 'PNEUMONIA1/'
        png_path2 = 'PNEUMONIA2/'
    elif class_num == 3:
        png_path = 'COVID/'
        png_path1 = 'COVID1/'
        png_path2 = 'COVID2/'
    ds = pydicom.read_file(trian_data_path + dir_name + '/' + dir2_name[0] + '/' + file_name[0])
    img = ds.pixel_array

    save_name = png_path + file_name[0].replace('.dcm', '.png')
    save_name1 = png_path1 + file_name[0].replace('.dcm', '.png')
    save_name2 = png_path2 + file_name[0].replace('.dcm', '.png')

    plt.imsave(save_name, img)
    time.sleep(2)
    img = Image.open(save_name).convert('L')
    img.save(save_name) ############ 1 gray
    time.sleep(2)

    image, mask = inference.inference(model,save_name, 0.2)
    temp_img = inference.img_with_masks( image, [mask[0], mask[1]], alpha = 0.1)
    plt.imsave(save_name1, image) ############ 2 mark
    image1 = image.copy()
    image2 = image.copy()
    mask[0] = np.where(mask[0]>0, 1, mask[0])
    mask[1] = np.where(mask[1]>0, 1, mask[1])
    image2[:,:,0] = image1[:,:,0]*mask[0] + image1[:,:,0]*mask[1]
    image2[:,:,1] = image1[:,:,1]*mask[0] + image1[:,:,1]*mask[1]
    image2[:,:,2] = image1[:,:,2]*mask[0] + image1[:,:,2]*mask[1]
    try:
        plt.imsave(save_name2, image2) ############ 3 lung
    except:
        plt.imsave(save_name2, normalize(image2)) ############ 3 lung
    time.sleep(2)


#%%

'''
資料處理後，即可將 pre-trained model 下去做訓練及預測
'''

normal = 'NORMAL'
PNEUMONIA = 'PNEUMONIA'
covid = 'COVID'
# Path list
dir_normal = os.listdir(normal)
dir_PNEUMONIA = os.listdir(PNEUMONIA)
dir_covid = os.listdir(covid)
'''
下面註解的地方只是將圖片印出來，可以不用跑
'''

# #%%
# mat.figure(figsize=(16,12))
# for i in range(6):
#     ran = random.choice((1,30))
#     normal1 = [os.path.join(normal, f) for f in dir_normal[ran:ran+1]]
#     rand = random.choice(normal1)
#     mat.subplot(3, 3, i+1)
#     img = mat.imread(rand)
#     mat.imshow(img,cmap = 'gray')
#     mat.axis(False)
#     mat.title('Normal X-ray images')
# mat.show()
# #%%
# mat.figure(figsize=(16,12))
# for i in range(6):
#     ran = random.choice((1,30))
#     covid1 = [os.path.join(covid, f) for f in dir_covid[ran:ran+1]]
#     rand = random.choice(covid1)
#     mat.subplot(3, 3, i+1)
#     img = mat.imread(rand)
#     mat.imshow(img,cmap = 'gray')
#     mat.axis(False)
#     mat.title('Covid X-ray images')
# mat.show()
# #%%
# mat.figure(figsize=(16,12))
# for i in range(6):
#     ran = random.choice((1,30))
#     covid1 = [os.path.join(PNEUMONIA, f) for f in dir_PNEUMONIA[ran:ran+1]]
#     rand = random.choice(covid1)
#     mat.subplot(3, 3, i+1)
#     img = mat.imread(rand)
#     mat.imshow(img,cmap = 'gray')
#     mat.axis(False)
#     mat.title('PNEUMONIA images')
# mat.show()
# #%%
# x = Augmentor.Pipeline("COVID")
# x.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
# x.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
# x.flip_left_right(probability=1)
# x.process()  
# x.sample(200) #generate 580 augmented images based on your specifications

#%%
'''
把圖片和label設定好
'''
imagePaths = list(paths.list_images(normal))
data = []
labels = []
for imagePath in imagePaths:
    label = 0
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
data = np.array(data) /255
labels = np.array(labels)

imagePaths1 = list(paths.list_images(covid))
data1 = []
labels1 = []
for imagePath in imagePaths1:
    label1 = 1
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data1.append(image)
    labels1.append(label1)
data1 = np.array(data1) / 255
labels1 = np.array(labels1)


imagePaths2 = list(paths.list_images(PNEUMONIA))
data2 = []
labels2 = []
for imagePath in imagePaths2:
    label2 = 2
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data2.append(image)
    labels2.append(label2)
data2 = np.array(data2) / 255
labels2 = np.array(labels2)
#%%
'''
以下開始訓練，baseModel可以換成喜歡的，檢查一下一開始有沒有import進來就可以了
'''

dataset = np.concatenate((data, data1, data2), axis=0)
label = np.concatenate((labels, labels1, labels2), axis=0)
label = to_categorical(label)
(trainX, testX, trainY, testY) = train_test_split(dataset, label, test_size=0.30, stratify=label, random_state=42)
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.30, random_state=42)
# initialize the initial learning rate, number of epochs, and batch size
INIT_LR = 1e-4
EPOCHS = 90
BS = 8

# baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# baseModel = VGG19(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# baseModel = DenseNet201(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
baseModel = NASNetLarge(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

callbacks = [EarlyStopping(monitor='val_loss', patience=8),ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

for layer in baseModel.layers:
    layer.trainable = False

# compile our model
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the head of the network
H = model.fit(trainX, trainY, validation_data=(valX,valY),batch_size = BS, epochs=EPOCHS,callbacks=callbacks)

#%%
'''
這邊也都不用跑，是之後做報告可以用的視覺化結果
'''
# acc = H.history['accuracy']
# loss = H.history['loss']
# val_loss = H.history['val_loss']
# val_acc = H.history['val_accuracy']
# epochs = range(len(H.epoch))

# title1 = 'Accuracy vs Validation Accuracy'
# leg1 = ['Acc', 'Val_acc']
# title2 = 'Loss vs Val_loss'
# leg2 = ['Loss', 'Val_loss']

# def plot(epochs, acc, val_acc, leg, title):
#     mat.plot(epochs, acc)
#     mat.plot(epochs, val_acc)
#     mat.title(title)
#     mat.legend(leg)
#     mat.xlabel('epochs')

# mat.figure(figsize=(15,5))
# mat.subplot(1,2,1)
# plot(epochs, acc, val_acc, leg1, title1)
# mat.subplot(1,2,2)
# plot(epochs, loss, val_loss, leg2, title2)
# mat.show()

# # make predictions on the testing set
# print("[INFO] evaluating network...")
# predIdxs = model.predict(testX, batch_size=BS)
# predIdxs = np.argmax(predIdxs, axis=1)
# print(classification_report(testY.argmax(axis=1), predIdxs,digits=4))

# # compute the confusion matrix and and use it to derive the raw
# # accuracy, sensitivity, and specificity
# import seaborn as sns
# cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
# sns.set(font_scale=1)#for label size
# sns.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 12})# font siz
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.figure(figsize = (16,12))
# total = sum(sum(cm))
# acc = (cm[0, 0] + cm[1, 1]) / total
# sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
# specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# # show the confusion matrix, accuracy, sensitivity, and specificity
# print("acc: {:.4f}".format(acc))
# print("sensitivity: {:.4f}".format(sensitivity))
# print("specificity: {:.4f}".format(specificity))

# ypred = model.predict(testX)
# total = 0
# accurate = 0
# accurateindex = []
# wrongindex = []

# for i in range(len(ypred)):
#     if np.argmax(ypred[i]) == np.argmax(testY[i]):
#         accurate += 1
#         accurateindex.append(i)
#     else:
#         wrongindex.append(i)
        
#     total += 1
    
# print('Total-test-data;', total, '\taccurately-predicted-data:', accurate, '\t wrongly-predicted-data: ', total - accurate)
# print('Accuracy:', round(accurate/total*100, 3), '%')
# label= {0: 'Normal', 1: 'Covid', 2: 'Pneumonia'}
# imidx = random.sample(accurateindex, k=9)# replace with 'wrongindex'

# nrows = 3
# ncols = 3
# fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))

# n = 0
# for row in range(nrows):
#     for col in range(ncols):
#             ax[row,col].imshow(testX[imidx[n]])
#             ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(label[np.argmax(ypred[imidx[n]])], label[np.argmax(testY[imidx[n]])]))
#             n += 1

# plt.show()
# %%
'''
產生submission，記得想用甚麼model要檢查一下baseModel有沒有用對模型~
'''
valid = 'VALID'
dir_valid = os.listdir(valid)

imagePaths3 = list(paths.list_images(valid))
data3 = []
question_name = []
for imagePath in imagePaths3:
    question_name.append(imagePath)
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data3.append(image)
data3 = np.array(data3) / 255

# baseModel = VGG19(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
baseModel = NASNetLarge(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
model.load_weights('best_model.h5')

ypred = model.predict(data3)
ans_list = []

for i in range(len(ypred)):
    ans_list.append(np.argmax(ypred[i]))

filename_list = [] 
for i in question_name:
    i = i.split('/')[1]
    i = i.split('.')[0]
    filename_list.append(i)

answwer_list = [] 
for i in ans_list:
    if i == 0:
        i = 'Negative'
    if i == 1:
        i = 'Atypical'
    if i == 2:
        i = 'Typical'
    answwer_list.append(i)
print(filename_list)
print(answwer_list)

output_file_name = 'submission3.csv'
dict = {'FileID':filename_list,'Type':answwer_list}
df = pd.DataFrame(dict) 
df.to_csv(output_file_name,index=None)
# %%
