!git clone https://bitbucket.org/jadslim/german-traffic-signs
!ls german-traffic-signs/
import keras
import numpy as np
import  matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import pickle
import pandas as pd
import random
with open('german-traffic-signs/train.p','rb') as f:
  train_data = pickle.load(f)

with open('german-traffic-signs/valid.p','rb') as f:
  val_data = pickle.load(f)

with open('german-traffic-signs/test.p','rb') as f:
  test_data = pickle.load(f)
  
X_train , y_train = train_data['features'], train_data['labels']
X_val , y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']
# FOR DEBUGGING PURPOSE
assert(X_train.shape[0] == y_train.shape[0]), "The num of lables doesnt match the number of images"
assert(X_test.shape[0] == y_test.shape[0]), "The num of lables doesnt match the number of images"
assert(X_val.shape[0] == y_val.shape[0]), "The num of lables doesnt match the number of images"

assert(X_train.shape[1:] == (32,32,3)), "The image is not 32x32x3"
assert(X_test.shape[1:] == (32,32,3)), "The image is not 32x32x3"
assert(X_val.shape[1:] == (32,32,3)), "The image is not 32x32x3"
data = pd.read_csv('german-traffic-signs/signnames.csv')
print(data)
# Lets visualize the data
num_samples = []
num_classes = 43
cols = 5

fig,axs = plt.subplots(nrows = num_classes , ncols = cols ,figsize=(10,50))
fig.tight_layout()

for i , row in data.iterrows():
  for j in range(cols):
    x = X_train[y_train == i]
    axs[i][j].imshow(x[random.randint(0,len(x)-1),:,:,:])
    axs[i][j].axis('off')
    
    if (j==2):
      axs[i][j].set_title(str(i)+ '-' + row['SignName'])
      num_samples.append(len(x))
      

plt.figure(figsize=(12,4))
plt.bar(range(0,num_classes),num_samples)
# Preprocessing
import cv2

plt.imshow(X_train[1000])
plt.axis('off')
print(X_train[1000].shape)
print(y_train[1000])

# Preprocessing technique 1 = grayscale conversion
# 2 reasons
# 1. The colors are not impot for classification since many
# traffic signals are essentially the same color
# 2. Converting to grayscale significantly reduce the computational
# power to process the image
def grayscale(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return gray

img = grayscale(X_train[1000])
plt.imshow(img)
plt.axis('off')
print(img.shape)

# Preprocessing technique 2
# Histogram equalization for similar lighting effect on all images
# This also leads to better contrast
def equalize(img):
  img = cv2.equalizeHist(img) # the function only accepts grayimg
  return img

img =  equalize(img)
plt.imshow(img)
plt.axis('off')
  
def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img/255 # preprocess step 3 normalization
  return img

X_train = np.array(list(map(preprocessing,X_train)))
X_test = np.array(list(map(preprocessing,X_test)))
X_val = np.array(list(map(preprocessing,X_val)))
plt.imshow(X_train[random.randint(0,len(X_train)-1)])
plt.axis('off')
print(X_train.shape)

# after we have pre-processed our images lets add depth to our images
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            rotation_range=10,
                            shear_range=0.1)

datagen.fit(X_train)
batches = datagen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)

fig,axs = plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()

for i in range(15):
  axs[i].imshow(X_batch[i].reshape(32,32))
  axs[i].axis('off')

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
# The final step to prepare our data is one-hot encoding
y_train = to_categorical(y_train,43)
y_test = to_categorical(y_test,43)
y_val = to_categorical(y_val,43)
def lenet():
  model = Sequential()
  model.add(Conv2D(60,(5,5),input_shape=(32,32,1),activation='relu'))
  model.add(Conv2D(60,(5,5),activation='relu'))
  model.add(MaxPooling2D(2,2))
  
  model.add(Conv2D(30,(3,3),activation='relu'))
  model.add(Conv2D(30,(3,3),activation='relu'))
  model.add(MaxPooling2D(2,2))
  
  
  model.add(Flatten())
  model.add(Dense(300,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes,activation='softmax'))
  model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
  return model
model = lenet()
h = model.fit_generator(datagen.flow(X_train,y_train,batch_size=50),steps_per_epoch=2000,epochs=10,validation_data=(X_val,y_val),shuffle=1)
score = model.evaluate(X_test,y_test,verbose=1)
print(score[0])
print(score[1])
print(X_test.shape)
img = X_test[35,:,:,:].reshape(32,32)
plt.imshow(img)
img = img.reshape(1,32,32,1)
print(img.shape)
pred = model.predict_classes(img)
print("The prediction is: ", str(pred))

#Things to do change the accuracy
#1. decrease the learning rate
#2. increasing the number of filters
#3. Adding more conv layers - This actually requires less computing power, because the with addition of more convolutional layers, we get 
#lesser parameters (counterintuitive)

#things to do decrease overfitting
#increase number of dropout layers

# The master weapon
#We noticed that our model wasnt able to classify our data perfectly
#So we do what is called as data augmentation, we change our images in some ways so as to obtain different perspectives of the
#same data


#lets test it on real data
import cv2
import requests
from PIL import Image
url = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
resp = requests.get(url,stream='true')
img = Image.open(resp.raw)
plt.imshow(img,cmap=plt.get_cmap('gray'))
plt.axis('off')
#preprocess
img = np.asarray(img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)
img = img/255
plt.imshow(img,cmap=plt.get_cmap('gray'))
plt.axis('off')

img = cv2.resize(img,(32,32))
plt.imshow(img,cmap = plt.get_cmap('gray'))
img = img.reshape(1,32,32,1)
print(img.shape)

pred = model.predict_classes(img)
print('prediction',str(pred))
