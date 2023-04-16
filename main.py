import os
import cv2

# To test the web-cam
capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

# Background of the image captured by the webcam
imgBackground = cv2.imread('Resources/background.png')

# Importing the images of each mode in a list
modesFolderPath = 'Resources/Modes'
modesImgNameList = os.listdir(modesFolderPath)
modesImgList = []
for imgName in modesImgNameList:
    modesImgList.append(cv2.imread(os.path.join(modesFolderPath, imgName)))
# end for

modeType = 0 # the current state of the attendance cam
while True:
    success, img = capture.read()
    
    # Overlaying the webcam-captured image on the background
    imgBackground[162: 162 + 480, 55: 55 + 640] = img
    # Overlaying the current mode image
    imgBackground[44:44 + 633, 808:808 + 414] = modesImgList[modeType]

    cv2.imshow("Face Attendance Cam Activated", imgBackground)
    cv2.waitKey(1)
# end while

# end main.py