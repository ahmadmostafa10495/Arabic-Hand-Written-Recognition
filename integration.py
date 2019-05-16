import os, os.path
import cv2
import numpy as np
#from keras.models import load_model


#Import CNN Model for character recognition
#char_model = load_model('CNN models/Arabic_OCR_model.h5')

#Import CNN Model for digit recognition
#digit_model = load_model('CNN models/Digit_OCR_model.h5')

output_string = ""


#make a dictionary for the Arabic Char from 0 - 27
ArLetters = {0:"ا",1:"ب",2:"ت",3:"ث",4:"ج",5:"ح",6:"خ",7:"د",8:"ذ",9:"ر",10:"ز",
            11:"س",12:"ش",13:"ص",14:"ض",15:"ط",16:"ظ",17:"ع",18:"غ",19:"ف",20:"ق",
            21:"ك",22:"ل",23:"م",24:"ن",25:"ه",26:"و",27:"ي"}

output_file = open("outputfile.txt" , "w" )
a=0
# for loop over the folders with the name 0 to 18

###should change path depending on pc.....probably get it as an input from user###
arrToCompWith = np.full((16,16), 255)
for fnum in range(19):
    stringToBeAdded = ""
    
    #changing directory
    os.chdir("F:\\GitHub\\Arabic-Handwritten-Recogonotion\\preprocessing\\test_case output\\"+str(fnum))
    #getting number of imgs
    imnum = len([name for name in os.listdir('.') if os.path.isfile(name)])
    # for loop over the images inside the folders
    for imgindex in range(imnum):
        # Pre processing for each image
        img = cv2.imread(str(imgindex)+'.png',1)
        
        # Convert to Gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #convert to binary
        new= img > (150)
        new = new.reshape(1, 32, 32,1).astype('float32')
    
        #check the image is all white and if yes put a " " in the output file
        rows = new.shape[0]
        cols = new.shape[1]
        #x = new[int(rows * 0.25):int(rows * 0.75),int(cols * 0.25):int(cols * 0.75)].astype('float32')
        imgToCompWith = img[8:24,8:24]
        x = np.bitwise_and(arrToCompWith,imgToCompWith)
        if np.equal(x, arrToCompWith).all():
            stringToBeAdded = stringToBeAdded + " "
            continue

        #y = loaded_model.predict_classes(new)
        # map the y to the coressponding char
        stringToBeAdded = stringToBeAdded + ArLetters[y]
        # in case of digit no need yo map




    output_file.write(stringToBeAdded+"\n")    
            






# Close opend file
output_file.close()
