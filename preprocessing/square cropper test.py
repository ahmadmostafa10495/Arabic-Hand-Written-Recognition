from __future__ import print_function
import os
import sys
import numpy as np
import cv2 as cv

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


def main():
    from glob import glob
    for fn in glob('metro11.jpg'):
        img = cv.imread(fn)
        cropped_images = crop_squares(img)

        for i in range(len(cropped_images)):
            for j in range(len(cropped_images[i])):
                print_image_on_folder(cropped_images[i][j], str(j), str(i))

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()