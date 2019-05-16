# import for the crop file

from __future__ import print_function
import os
import sys
import numpy as np
import cv2

# imports for the predict file
import os, os.path
import numpy as np
from keras.models import load_model
from keras import backend as K
import collections
K.set_image_dim_ordering('th')
import matplotlib.pyplot as plt



PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def draw_squares(img, squares):
    cv.drawContours(img, squares, -1, (0, 255, 0), 3)
    cv.imshow('squares', img)
    cv.imwrite('result.png', img)
    cv.waitKey()


def crop_small_squares(img, big_square_center):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    centers = []
    images =[]
    squares_img = np.zeros_like(img)
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                __retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and (cv.contourArea(cnt) > 20000 and cv.contourArea(cnt) < 45000)and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.2:
                        center = [int((cnt[0][1] + cnt[1][1]+ cnt[2][1]+cnt[3][1])/4), int(abs((cnt[0][0] + cnt[1][0]+cnt[2][0]+cnt[3][0])/4))]

                        x_start = min(cnt[0][1], cnt[1][1], cnt[2][1], cnt[3][1])
                        x_end = max(cnt[0][1], cnt[1][1], cnt[2][1], cnt[3][1])
                        y_start = min(cnt[0][0], cnt[1][0], cnt[2][0], cnt[3][0])
                        y_end = max(cnt[0][0], cnt[1][0], cnt[2][0], cnt[3][0])

                        if squares_img[center[0]][center[1]][1] == 0:
                            squares.append(cnt)
                            centers.append(center)
                            cropped_img = crop_square(img, cnt)
                            cropped_img_resized = cv.resize(cropped_img, (32, 32))
                            images.append(cropped_img_resized)
                            #print_image_on_folder(cropped_img1, str(center[1]), str(big_square_center))
                            squares_img = make_all_ones(squares_img, center, x_start, x_end, y_start, y_end)
    #print(centers)
    x,y = zip (*centers)
    y, images = (list(t) for t in zip(*sorted(zip(y, images), reverse=True)))
    return images


def crop_square(img,square):
    polygon = [square]
    # First find the minX minY maxX and maxY of the polygon
    minX = img.shape[1]
    maxX = -1
    minY = img.shape[0]
    maxY = -1
    for point in polygon[0]:

        x = point[0]
        y = point[1]

        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y

    # Go over the points in the image if thay are out side of the emclosing rectangle put zero
    # if not check if thay are inside the polygon or not
    cropedImage = np.zeros_like(img)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):

            if x < minX or x > maxX or y < minY or y > maxY:
                continue

            if cv.pointPolygonTest(np.asarray(polygon), (x, y), False) >= 0:
                cropedImage[y, x, 0] = img[y, x, 0]
                cropedImage[y, x, 1] = img[y, x, 1]
                cropedImage[y, x, 2] = img[y, x, 2]

    # Now we can crop again just the envloping rectangle

    finalImage = cropedImage[minY:maxY, minX:maxX]
    #print_image_on_folder(finalImage,img_name,folder_name)
    return finalImage


def print_image_on_folder(img, img_name, folder_name):
    path = './' + folder_name
    if os.path.exists(path) is False:
        os.mkdir(path)
    #os.makedirs(folder_name)
    path = path + '/' + img_name + '.png'
    cv.imwrite(path, img)


def make_all_ones(squares_img, center, x_start, x_end, y_start, y_end):
    squares_img[center[0]][center[1]] = [1, 1, 1]
    for i in range(x_start , x_end+1):
        squares_img[i][y_start:y_end+1] = [1, 1, 1]

    return squares_img


def crop_squares(img):
    img = cv.resize(img, (3500, 6300))
    img = cv.GaussianBlur(img, (5, 5), 0)
    large_squares = []
    small_squares = []
    centers = []
    squares_img = np.zeros_like(img)
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                __retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 50000 and cv.contourArea(
                        cnt) < 10000000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        center = [int((cnt[0][1] + cnt[1][1]+ cnt[2][1]+cnt[3][1])/4), int(abs((cnt[0][0] + cnt[1][0]+cnt[2][0]+cnt[3][0])/4))]

                        x_start = min(cnt[0][1], cnt[1][1], cnt[2][1], cnt[3][1])
                        x_end = max(cnt[0][1], cnt[1][1], cnt[2][1], cnt[3][1])
                        y_start = min(cnt[0][0], cnt[1][0], cnt[2][0], cnt[3][0])
                        y_end = max(cnt[0][0], cnt[1][0], cnt[2][0], cnt[3][0])

                        if squares_img[center[0]][center[1]][1] == 0:
                            large_squares.append(cnt)
                            centers.append(center)
                            cropped_img = crop_square(img, cnt)
                            small_squares.append(crop_small_squares(cropped_img, str(center)))
                            squares_img = make_all_ones(squares_img, center, x_start, x_end, y_start, y_end)
    centers, large_squares, small_squares = (list(t) for t in zip(*sorted(zip(centers, large_squares, small_squares))))
    draw_squares(img, large_squares)
    return small_squares




############################################################################







def read_and_predict():
    # Import CNN Model for character recognition
    char_model = load_model("C:\\Users\\dell\\Desktop\\Image_project_integration\\CNN_models\\Arabic_OCR_model.h5")
    digit_model = load_model("C:\\Users\\dell\\Desktop\\Image_project_integration\\CNN_models\\new_digits_model.h5")

    # model.load_weights("C:\\Users\\dell\\Desktop\\Image_project_integration\\CNN_models\\weights_file.h5")

    output_string = ""

    # make a dictionary for the Arabic Char from 0 - 37
    ArLetters = {0: "ا", 1: "ب", 2: "ت", 3: "ث", 4: "ج", 5: "ح", 6: "خ", 7: "د", 8: "ذ", 9: "ر", 10: "ز",
                 11: "س", 12: "ش", 13: "ص", 14: "ض", 15: "ط", 16: "ظ", 17: "ع", 18: "غ", 19: "ف", 20: "ق",
                 21: "ك", 22: "ل", 23: "م", 24: "ن", 25: "ه", 26: "و", 27: "ي", 28: "0", 29: "1", 30: "2", 31: "3",
                 32: "4", 33: "5",
                 34: "6", 35: "7", 36: "8", 37: "9"}
    output_file = open("C:\\Users\\dell\\Desktop\\Image_project_integration\\output.txt", "w", encoding='utf-8')
    a = 0
    # for loop over the folders with the name 0 to 18

    ###should change path depending on pc.....probably get it as an input from user###
    arrToCompWith = np.full((16, 16), 255)

    images = []

    for fnum in range(19):
        stringToBeAdded = ""
        flag_digit = False

        # changing directory
        os.chdir("C:\\Users\\dell\\Desktop\\Image_project_integration\\test_case_2output\\" + str(fnum))
        # getting number of imgs
        imnum = len([name for name in os.listdir('.') if os.path.isfile(name)])
        # for loop over the images inside the folders
        number_line = ""
        for imgindex in range(imnum):
            # Pre processing for each image
            img = cv2.imread(str(imgindex) + '.png', 1)
            # Convert to Gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # check the image is all white and if yes put a " " in the output file
            rows = img.shape[0]
            cols = img.shape[1]
            # x = new[int(rows * 0.25):int(rows * 0.75),int(cols * 0.25):int(cols * 0.75)].astype('float32')
            imgToCompWith = img[8:24, 8:24]
            x = np.bitwise_and(arrToCompWith, imgToCompWith)
            if np.equal(x, arrToCompWith).all():
                stringToBeAdded = stringToBeAdded + " "
                continue

            # resize image
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

            # reshape image
            img = img.reshape(1, 32, 32, 1).astype('float32')

            # convert to binary
            img = img > (150)

            queue = collections.deque()

            # in case of digit no need to map
            if fnum == 3 or fnum == 4 or fnum == 5 or fnum == 6 or fnum == 7 or fnum == 8 or fnum == 12 or fnum == 13 or fnum == 14 or fnum == 16 or fnum == 18:
                ##digit model
                temp = str(digit_model.predict_classes(img)[0])
                number_line += temp
                flag_digit = True

                img_show = img.reshape(32, 32).astype('float32')
                images.append(img_show)


            else:
                ##letter model
                y = char_model.predict_classes(img)[0]
                stringToBeAdded = stringToBeAdded + ArLetters[y]

            if (flag_digit):
                stringToBeAdded = number_line[::-1]

        output_file.write(stringToBeAdded + "\n")


        # Close opend file
    output_file.close()


def main():
    from glob import glob
    for fn in glob('metro1.jpg'):
        img = cv.imread(fn)
        cropped_images = crop_squares(img)

        for i in range(len(cropped_images)):
            for j in range(len(cropped_images[i])):
                print_image_on_folder(cropped_images[i][j], str(j), str(i))

    print('Done')
    read_and_predict()



if __name__ == '__main__':
    print(__doc__)
    main()