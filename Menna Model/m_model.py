from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import cv2
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, array_to_img
# import padas as pd
from PIL import Image
import glob
import matplotlib.pyplot as plt

training_data = []
training_labels = []
imagePath = "E:\\GitHub\\Arabic-Handwritten-Recogonotion\\train/*.png"

for filename in glob.glob(imagePath): #assuming gif

    im=cv2.imread(filename)
    im = cv2.resize(im, (32, 32))

    image = img_to_array(im)
    training_data.append(image)
    label1 = filename.split("_")
    label2 = label1[3].split(".")
    training_labels.append(label2[0])

training_labels = np_utils.to_categorical(training_labels)
training_labels = training_labels[:, 1:]




test_data = []
test_labels = []


imagePath2 = "E:\\GitHub\\Arabic-Handwritten-Recogonotion\\test/*.png"

for filename2 in glob.glob(imagePath2): #assuming gif

    im2=cv2.imread(filename2)
    im2 = cv2.resize(im2, (32, 32))

    image2 = img_to_array(im2)
    test_data.append(image2)
    label1 = filename2.split("_")
    label2 = label1[3].split(".")
    test_labels.append(label2[0])


test_labels= np_utils.to_categorical(test_labels)
test_labels = test_labels[:, 1:]

# dataset = np.loadtxt('E:\\GitHub\\Arabic-Handwritten-Recogonotion\\csvTestLabel_3360x1.csv', delimiter=',', skiprows=1)
# for i in range(336):
#     # load the image, pre-process it, and store it in the data list
#
#     imagePath = "E:\\GitHub\\Arabic-Handwritten-Recogonotion\\test"
#     imageName = "\\id_" + str(i) +
#     image = cv2.imread(imagePath)
#
#     image = img_to_array(image)
#     data.append(image)
#
#     # extract the class label from the image path and update the
#     # labels list
#     label = imagePath.split(os.path.sep)[-2]
#     label = 1 if label == "santa" else 0
#     labels.append(label)





model = Sequential()
model.add(Conv2D(80 , (5 ,5), activation = 'relu', input_shape = (32 , 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64 , (5 ,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(28, activation='softmax'))
model.compile(loss ='categorical_crossentropy',
              optimizer = Adam () ,
              metrics=['accuracy'])

# model.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)

history = model.fit(np.array(training_data)/255, np.array(training_labels), epochs=150, batch_size=210 , validation_data = (np.array(test_data)/255, np.array(test_labels)))
# score = model.evaluate(test_data, test_labels, batch_size=210, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save('E:\\GitHub\\Arabic-Handwritten-Recogonotion\\my_model.h5')  # creates a HDF5 file 'my_model.h5'

#pre = model.predict_classes(test_data[0])

