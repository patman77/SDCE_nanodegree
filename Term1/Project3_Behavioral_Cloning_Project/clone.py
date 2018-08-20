import csv
import cv2
import numpy as np
#import keras

lines = []
#with open('./data/driving_log.csv') as csvfile:
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        print(line)
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    path = source_path.split('/')[0]
    #print('source_path ', source_path, ', path ', path)
    current_path = './data/IMG/' + filename
    #print("current_path ", filename)
    image =cv2.imread(current_path)
    images.append(image)
    #print('shape img ', image.shape)
    measurement = float(line[3])
    measurements.append(measurement)

imagesasarray = np.asarray(images)

X_train = np.array(imagesasarray)
print('shape X_train: ', X_train.shape)
print('shape imagesasarray: ', imagesasarray.shape)
y_train = np.array(measurements)
print('shape y_train: ', y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20)

model.save('model.h5')
