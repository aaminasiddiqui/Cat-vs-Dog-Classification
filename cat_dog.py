import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

#Data augmentation

train_datagen=ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

#Generators
batch_size=12
train_generator=train_datagen.flow_from_directory(
    '/content/drive/MyDrive/CATS_DOGS/train',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator=test_datagen.flow_from_directory(
    '/content/drive/MyDrive/CATS_DOGS/test',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary'
)

#CNN model 32,64,128
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

results=model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator
)

import matplotlib.pyplot as plt

plt.plot(results.history['accuracy'],color='red',label='train')
plt.plot(results.history['val_accuracy'],color='blue',label='test')
plt.legend()
plt.show()

from keras.models import load_model

model.save('catdog.h5')
model1=load_model('catdog.h5')

from tensorflow.keras.preprocessing import image
import numpy as np

cat_img=image.load_img('cat.jpg',target_size=(150,150))
cat_img=image.img_to_array(cat_img)
cat_img=np.expand_dims(cat_img,axis=0)
cat_img=cat_img/255

prediction=newModel.predict(cat_img)
classification=np.argmax(newModel.predict(cat_img), axis=-1)

print(classification)