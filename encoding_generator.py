import cv2
import os
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate("facial-recog-attendance-firebase-adminsdk-m0o07-0b6db63d1f.json")
firebase_admin.initialize_app(cred, {
    "databaseURL" : "https://facial-recog-attendance-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "storageBucket" : "facial-recog-attendance.appspot.com"
})


# Import student images
studentImgFolderPath = 'Images'
studentImgNameList = os.listdir(studentImgFolderPath)
studentImgList = []
studentIDList = []
for imgName in studentImgNameList:
    studentImgList.append(cv2.imread(os.path.join(studentImgFolderPath,
                                                    imgName)))
    studentIDList.append(os.path.splitext(imgName)[0])

    # Send the images to the firebase database
    # image_path stores the path of each particular image
    image_path = f"{studentImgFolderPath}/{imgName}"
    bucket = storage.bucket()
    blob = bucket.blob(image_path)
    blob.upload_from_filename(image_path)
# end for


def findEncodings(imgList):
    encodingList = []
    for img in imgList:
        # Converting to RGB format as face_recognition module works only
        # with RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodingList.append(encoding)
    # end for

    return(encodingList)
# end function findEncodings()

# Create encodings of the "known" faces (i. e., the faces already saved
# in the database)
print("Encoding Started (for known faces)...")
known_EncodingList = findEncodings(studentImgList) # function call
print("Encoding Successful!")


# Save the encodings with their respective IDs in a pickle file
# so that it can be imported in main.py
known_EncodingWithIDsList = [known_EncodingList, studentIDList]

# Create a file where pickle file is to be dumped
print("Saving the encodings in file...")
file = open("encoding_with_IDs.p", "wb")
pickle.dump(known_EncodingWithIDsList, file)
file.close()
print("File saved successfully!")

# end encoding_generator.py