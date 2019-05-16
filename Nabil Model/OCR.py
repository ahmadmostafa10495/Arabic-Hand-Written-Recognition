import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils.np_utils import to_categorical
import cv2

train_x = []
train_y = []
j = 1

# for i in range(13440):
#     a = "train/id_"+str(i+1)+"_label_"+str(j)+".png"
#     train_x.append(cv2.imread(a))
#     train_y.append(j)
#     if (i+1) % 8 == 0:
#         if j+1 == 29:
#             j = 1
#         else:
#             j = j + 1
#
# train_x = np.array(train_x)
# train_y = np.array(train_y)
#
# train_y = to_categorical(train_y)
# train_y = train_y[:,1:]
#
# train_x = train_x/255
# train_x = train_x[:,:,:,1]
# train_x = train_x.reshape(train_x.shape[0], 32, 32, 1).astype('float32')

# print(train_x[0])

# validate_x = []
# validate_y = []
# j = 1
#
# for i in range(12097,13440):
#     a = "train/id_"+str(i+1)+"_label_"+str(j)+".png"
#     validate_x.append(cv2.imread(a))
#     validate_y.append(j)
#     if (i+1) % 8 == 0:
#         if j+1 == 29:
#             j = 1
#         else:
#             j = j + 1
#
# validate_x = np.array(validate_x)
# validate_y = np.array(validate_y)


test_x = []
test_y = []
j = 1
# print(train_x.shape)
# for i in range (3360):
#     a = "test/id_"+str(i+1)+"_label_"+str(j)+".png"
#     test_x.append(cv2.imread(a))
#     test_y.append(j)
#     if (i+1) % 2 == 0:
#         if j+1 == 29:
#             j = 1
#         else:
#             j = j + 1
#
# test_x = np.array(test_x)
# test_y = np.array(test_y)
#
# test_y = to_categorical(test_y)
# test_y = test_y[:,1:]
#
# test_x = test_x/255
#
# test_x = test_x[:,:,:,1]
# test_x = test_x.reshape(test_x.shape[0],32,32,1)

# print(train_x[0][10])

model = Sequential()
conv1 = model.add(Conv2D(80, (5, 5), activation='relu', input_shape=(32, 32, 1)))
pool1 = model.add(MaxPooling2D(pool_size=(2, 2)))
drop1 = model.add(Dropout(0.5))
norm1 = model.add(BatchNormalization())

conv2 = model.add(Conv2D(64, (5, 5), activation='relu'))

pool2 = model.add(MaxPooling2D(pool_size=(2, 2)))
drop2 = model.add(Dropout(0.5))
norm2 = model.add(BatchNormalization())

model.add(Flatten())
fc1 = model.add(Dense(1024, activation='relu'))
drop3 = model.add(Dropout(0.5))
norm3 = model.add(BatchNormalization())
fc2 = model.add(Dense(28, activation='softmax'))

model.load_weights('weights_file.h5')

model.save('OCR_model.h5')


adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)

model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

# model.summary()


# history = model.fit(train_x, train_y, batch_size = 200, epochs = 200, shuffle = True, verbose = 1, validation_split = 0.1)

# score = model.evaluate(test_x, test_y, batch_size = 200)
# print(100-score[1]*100)

predict_x = []
for i in range(28):
    if i == 17 or i == 18:
        continue
    else:
        a = 'Belly/'+str(i+1)+".PNG"

    # print(a)
    k = cv2.imread(a)
    if i == 0:
        print(k)

        cv2.imwrite('b.png',k)
    k = cv2.threshold(k,127,255,cv2.THRESH_BINARY)

    k = k[1][0]
    if i == 0:
        print(k)

        cv2.imwrite('c.png',k)


    # cv2.imwrite('c.png',predict_x[0])
    k = cv2.resize(k,(32,32),interpolation = cv2.INTER_AREA)
    # print(k)
    # cv2.imwrite('d.png',predict_x[0])

    if i == 0:
        print(k)

        cv2.imwrite('d.png',k)



    predict_x.append(k)

    # print(k)
    # predict_x.append(k)

    # k = np.array(k > 127)
    # k = int (k)
    #

    # k = mask * k





predict_x = np.array(predict_x)
# print(predict_x.shape)

predict_x = predict_x / 255

print(predict_x.shape)

# predict_x = cv2.resize(predict_x,(32,32,1),interpolation = cv2.INTER_AREA)


# predict_x = predict_x[:,:,:,1]


# predict_x = predict_x / 255
# predict_x = predict_x[:,:,:,1]
# predict_x = predict_x.reshape(predict_x.shape[0],32,32,1).astype('float32')


# for i in predict_x:
#
#     print(model.predict(i))
predict_x = predict_x.reshape(predict_x.shape[0],32,32,1)
cv2.imwrite('k.png',predict_x[0])
a = model.predict_classes(predict_x)[0]
print(a)
# sample = open('weights file.txt','w')
# print(model.get_weights(), file = sample)
# sample.close()

# print(model.get_weights().shape)
# model.save_weights('weights_file.h5')



# import matplotlib.pyplot as plt
#
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
