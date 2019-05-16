import cv2

j = 1
for i in range (13440):
    a = "../train/id_"+str(i+1)+"_label_"+str(j)+".png"
    img = cv2.imread(a)
    img = cv2.bitwise_not(img)
    cv2.imwrite("id_"+str(i+1)+"_label_"+str(j)+".png",img)
    if (i+1) % 8 == 0:
        if j+1 == 29:
            j = 1
        else:
            j = j + 1
