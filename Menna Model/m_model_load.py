from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img


#########################



img = cv2.imread('C:\\Users\\MENA\\Desktop\\z.png')
#img = cv2.imread('E:\\GitHub\\Arabic-Handwritten-Recogonotion\\train\\id_211_label_27.png')

#(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

print('Original Dimensions : ', img.shape)

width = 32
height = 32
dim = (width, height)

#resized = cv2.resize(im_bw, dim, interpolation=cv2.INTER_AREA)

#resized = 1 - resized

resized = cv2.resize(img, (32, 32))
image = img_to_array(resized)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()




#######################

print("hello1")

model = load_model('E:\\GitHub\\Arabic-Handwritten-Recogonotion\\my_model.h5')

print("hello2")

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print("hello3")
# img = cv2.imread('test.jpg')
# img = cv2.resize(img,(320,240))

resized = np.reshape(resized,[1, 32, 32, 3])
img2 = np.array(resized)/255
img2 = 1-img2
print(img2)






classes = model.predict_classes(img2)

print (classes)