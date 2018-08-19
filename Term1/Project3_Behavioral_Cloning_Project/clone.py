import csv
import cv2
import numpy as np
#import keras

lines = []
#with open('./data/driving_log_stripped.csv') as csvfile:
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        print(line)
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    #print("current_path ", filename)
    image =cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

imagesasarray = np.asarray(images)

X_train = np.array(imagesasarray)
print('shape X_train: ', X_train.shape)
y_train = np.array(measurements)
print('shape y_train: ', y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')