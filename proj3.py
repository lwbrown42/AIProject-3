from PIL import Image
from sklearn import svm
from sklearn import cross_validation
import numpy as np
import sys
import os
import itertools

#Paths for the training sets
SMILEY_PATH = "../TRAINING/01/"
HAT_PATH = "../TRAINING/02/"
HASH_PATH = "../TRAINING/03/"
HEART_PATH = "../TRAINING/04/"
DOLLAR_PATH = "../TRAINING/05/"

#Function: processImage()
#Input: A string containing a file path to an image
#Output: The normalized, flattened array for the image
def processImage(image):

    im = Image.open(image)
    imageArray = list(im.getdata())

#    flattenedImageArray = list(itertools.chain(*imageArray))
    flattenedImageArray = [item for tempList in imageArray for item in tempList]

    #normalize the values
    for i in range(len(flattenedImageArray)):
        flattenedImageArray[i] = flattenedImageArray[i]/255

    return flattenedImageArray

def main(argv):

#    myImage = "../11.jpg"
    myImage = argv[1]

    #prepping some stuff
    directories = [SMILEY_PATH, HAT_PATH, HASH_PATH, HEART_PATH, DOLLAR_PATH]
    totalArray = []
    classifierArray = []

    #process the images?
    for path in directories:
        for fileName in os.listdir(path):
            tempArray = processImage(path + fileName)
            
            if path == SMILEY_PATH:
                classifierArray.append("smile")
            elif path == HAT_PATH:
                classifierArray.append("hat")
            elif path == HASH_PATH:
                classifierArray.append("hash")
            elif path == HEART_PATH:
                classifierArray.append("heart")
            elif path == DOLLAR_PATH:
                classifierArray.append("dollar")

            totalArray.append(tempArray)

    # load up and prep our image we're comparing
    myImageArray = processImage(myImage)
    myImageArray = np.array(myImageArray).reshape(1,-1)
    
#    X_train, X_test, y_train, y_test = cross_validation.train_test_split(totalArray,  classifierArray, test_size=.05, random_state=0)
   
    #get our SVM ready
    mySvm = svm.SVC(class_weight = "auto")
    mySvm.fit(totalArray, classifierArray)


#OOOOOH YEAH BROTHER GOT SOME TESTING THINGS UP IN HERE
#
#   mySvm.fit(X_train, y_train)
#
# for a little cross validation  
#    print("Accuracy: " + str(mySvm.score(X_test, y_test)))
#
# for a lot of cross validation?
#    scores = cross_validation.cross_val_score(mySvm, totalArray, classifierArray, cv=5)
#    print(scores)


    #DO THE THINGGGGGGGGGGGGGGG
    print(mySvm.predict(myImageArray)[0])
    

main(sys.argv)
