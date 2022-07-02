import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np
%matplotlib inline



blueImgPng = cv2.imread('DATA/main/TRAIN/blue+color/100.png')
blueImgJpeg = cv2.imread('DATA/main/TRAIN/blue+color/101.jpeg')
type(blueImgJpeg)



blueImgGif = imageio.mimread('DATA/main/TRAIN/blue+color/176.gif')
blueImgGif = np.array(blueImgGif)
blueImgGif = np.squeeze(blueImgGif, axis=0)
blueImgGif.shape



blueImgPng = cv2.cvtColor(blueImgPng, cv2.COLOR_BGR2RGB)
plt.imshow(blueImgPng)



blueImgJpeg = cv2.cvtColor(blueImgJpeg, cv2.COLOR_BGR2RGB)
plt.imshow(blueImgJpeg)



blueImgGif = cv2.cvtColor(blueImgGif, cv2.COLOR_BGR2RGB)
plt.imshow(blueImgGif)



from keras.preprocessing.image import ImageDataGenerator



image_gen = ImageDataGenerator(horizontal_flip=True,
                              vertical_flip=True,
                              zoom_range=0.2)
                              
                              
                              
plt.imshow(image_gen.random_transform(blueImgJpeg))



image_gen.flow_from_directory('DATA/main/TRAIN')



image_gen.flow_from_directory('DATA/main/TEST')



# Create the model



from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D



image_shape = (250,250,3)

model = Sequential()

# reduce layers?
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# try smaller dropout rate
model.add(Dropout(0.3))

model.add(Dense(11))
# try softmax
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
              
              
model.summary()



# Train the model



# increase batch size to 128 bc its standard?
# 4830 / 128 = 37.7
batch_size = 128

train_image_gen = image_gen.flow_from_directory('DATA/main/TRAIN',
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='categorical')
                                                
                                                

test_image_gen = image_gen.flow_from_directory('DATA/main/TEST',
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='categorical')
                                                
                                                
                                                
print(f"train: {train_image_gen.class_indices}")
print(f"test: {test_image_gen.class_indices}")



import warnings
warnings.filterwarnings('ignore')



# value of steps per epoch should be total number of samples / batch size
results = model.fit_generator(train_image_gen,epochs=4,
                              steps_per_epoch=38,
                              validation_data=test_image_gen,
                              validation_steps=12)
