import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage
from datetime import datetime as dt

cred = credentials.Certificate("facial-recog-attendance-firebase-adminsdk-m0o07-0b6db63d1f.json")
firebase_admin.initialize_app(cred, {
    "databaseURL" : "https://facial-recog-attendance-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "storageBucket" : "facial-recog-attendance.appspot.com"
})
# Create a storage bucket
bucket = storage.bucket()


# Background of the image captured by the webcam
imgBackground = cv2.imread('Resources/background.png')

# Import the images of each mode in a list
modesFolderPath = 'Resources/Modes'
modesImgNameList = os.listdir(modesFolderPath)
modesImgList = []
for imgName in modesImgNameList:
    modesImgList.append(cv2.imread(os.path.join(modesFolderPath, imgName)))
# end for


# Load the encodings of the known faces (saved in the database)
print("Loading encodings of known faces...")
file = open("encoding_with_IDs.p", "rb")
known_encodingWithIDsList = pickle.load(file)
file.close()
known_encodingList, studentIDList = known_encodingWithIDsList
print("Loaded encodings successfully!")

# Set-up the web-cam
capture = cv2.VideoCapture(0) # zero for default camera
capture.set(3, 640) # 3 -> CAP_PROP_FRAME_WIDTH
capture.set(4, 480) # 4 -> CAP_PROP_FRAME_HEIGHT
# Intialise variables
modeType = 0 # current state of the attendance cam
frameCount = 0 # track number of frames captured and time to be waited out
studentID = -1 # student id whose face matched
studentImgList = []

while True:
    success, img = capture.read()
    
    # Overlay the webcam-captured image on the background
    imgBackground[162: 162 + 480, 55: 55 + 640] = img
    # Overlay the current mode image
    imgBackground[44: 44 + 633, 808: 808 + 414] = modesImgList[modeType]

    # Prepare the image from webcam for commparisons
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25) # down-scaling image
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Use face_recognition module to locate the faces
    webCamFaceLoc = face_recognition.face_locations(imgSmall)
    webCamFaceEncoding = face_recognition.face_encodings(imgSmall,
                                                         webCamFaceLoc)
    
    if webCamFaceLoc:
        # Compare the web cam face-encodings with the saved encodings
        for faceEncode, faceLoc in zip(webCamFaceEncoding, webCamFaceLoc):
            faceMatches = face_recognition.compare_faces(known_encodingList,
                                                        faceEncode)
            faceDists = face_recognition.face_distance(known_encodingList,
                                                        faceEncode)

            # Get the index of the least distant face from the the database
            matchIndex = np.argmin(faceDists)

            # If captured face matches a face in database
            if faceMatches[matchIndex]:
                # Create fancy rectangle corners to  mark the face location
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                studentID = studentIDList[matchIndex] # save id of matched face

                if frameCount == 0:
                    frameCount = 1 # download data from database in 1st frame
                    modeType = 1 # face matched; "showing student details" mode
                # end if
            else:
                print("New face detected (not in database)")
            # end if-else
        # end for

        if frameCount != 0:
            # Download data in first frame captured
            if frameCount == 1:
                studentInfo = db.reference(f'Students/{studentID}').get()

                # Get image of the matched face from the database
                blob = bucket.get_blob(f'Images/{studentID}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # Update the attendance data of the student
                datetimeObj = dt.strptime(studentInfo['last_attendance_time'],
                                                    "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (dt.now() - datetimeObj).total_seconds()
                print(secondsElapsed)

                # if 20 s passed, web cam takes attendance again
                if secondsElapsed >= 20:
                    ref = db.reference(f'Students/{studentID}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(
                        studentInfo['total_attendance']
                    )
                    ref.child('last_attendance_time').set(
                        dt.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                else:
                    modeType = 3 # keep in "already_marked" state
                    frameCount = 0
                    imgBackground[44: 44 + 633, 808: 808 +
                                  414] = modesImgList[modeType]
                # end if-else
                
                print(studentInfo)
            # end if

            # If number of frames captured in (10, 20], change mode to "marked"
            if 10 < frameCount <= 20:
                modeType = 2
            # end if

            # Overlay "marked" mode
            imgBackground[44: 44 + 633, 808: 808 + 414] = modesImgList[modeType]

            if modeType != 3:
                if frameCount <= 10:
                    # Overlay student information on the mode image 1
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']),
                                (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['department']),
                                (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentID), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['grade']),
                                (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['semester']),
                                (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['class_of']),
                                (1125, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (100, 100, 100), 1)
                    
                    # The name should be centralised for the "name" label
                    (w, h), _ = cv2.getTextSize(studentInfo['name'],
                                                cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfo['name']),
                                (808 + offset, 445), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (50, 50, 50), 1)
                    
                    imgBackground[175: 175 + 216, 909: 909 + 216] = imgStudent

                
                    frameCount += 1
                # end if

                    # Reset the webcam for next attendance after
                    # 20 frames captured
                    if frameCount > 20:
                        frameCount = 0
                        modeType = 0
                        studentInfo = []
                        imgStudent = []
                        # Overlay current modetype
                        imgBackground[44: 44 + 633, 808: 808 +
                                      414] = modesImgList[modeType]
                    # end if
            # end if
        # end if
    else:
        modeType = 0
        frameCount = 0
        studentInfo = []
        imgStudent = []
    # end if-else

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
# end while

# end main.py