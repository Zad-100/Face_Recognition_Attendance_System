import os
import cv2
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage
from datetime import datetime as dt

cred = credentials.Certificate("facial-recog-attendance-firebase-adminsdk-m0o07-0b6db63d1f.json")
firebase_admin.initialize_app(cred, {
    "databaseURL" : "https://facial-recog-attendance-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "storageBucket" : "facial-recog-attendance.appspot.com"
})
# Creating a storage bucket
bucket = storage.bucket()

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

# Load the encodings of the known faces (saved in the database)
print("Loading encodings of known faces...")
file = open("encoding_with_IDs.p", "rb")
known_encodingWithIDsList = pickle.load(file)
file.close()
known_encodingList, studentIDList = known_encodingWithIDsList
print("Loaded encodings successfully!")

modeType = 0 # the current state of the attendance cam
frameCount = 0 # to track number of frames captured and time to be waited out
studentID = -1 # initialising the student id whose face matched
studentImgList = []
while True:
    success, img = capture.read()
    
    # Overlaying the webcam-captured image on the background
    imgBackground[162: 162 + 480, 55: 55 + 640] = img
    # Overlaying the current mode image
    imgBackground[44:44 + 633, 808:808 + 414] = modesImgList[modeType]

    # Preparing the image from webcam for commparisons
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25) # down-scaling image
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Using face_recognition module to locate the faces
    webCamFaceLoc = face_recognition.face_locations(imgSmall)
    webCamFaceEncoding = face_recognition.face_encodings(imgSmall,
                                                         webCamFaceLoc)

    # Comparing the web cam face-encodings with the saved encodings
        for faceEncode, faceLoc in zip(webCamFaceEncoding, webCamFaceLoc):
            faceMatches = face_recognition.compare_faces(known_encodingList,
                                                        faceEncode)
            faceDists = face_recognition.face_distance(known_encodingList,
                                                        faceEncode)

            # Getting the index of the least distant face from the the database
            matchIndex = np.argmin(faceDists)

            # If captured face matches a face in database
            if faceMatches[matchIndex]:
                # Creating fancy rectangle corners to  mark the face location
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                studentID = studentIDList[matchIndex] # fetch id of matched face

                if frameCount == 0:
                    frameCount = 1 # download data from database in 1st frame
                    modeType = 1 # face matched; "showing student details" mode
                # end if
            else:
                print("New face detected (not in database)")
            # end if-else
        # end for

        if frameCount != 0:
            # Downloading data in first frame captured
            if frameCount == 1:
                studentInfo = db.reference(f'Students/{studentID}').get()

                # Get image of the matched face from the database
                blob = bucket.get_blob(f'Images/{studentID}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Face Attendance Cam Activated", imgBackground)
    cv2.waitKey(1)
# end while

# end main.py