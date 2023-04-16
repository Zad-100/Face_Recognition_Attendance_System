import cv2
import os
import face_recognition
import pickle



# Importing student images
studentImgFolderPath = 'Images'
studentImgNameList = os.listdir(studentImgFolderPath)
studentImgList = []
studentIDList = []
for imgName in studentImgNameList:
    studentImgList.append(cv2.imread(os.path.join(studentImgFolderPath,
                                                    imgName)))
    studentIDList.append(os.path.splitext(imgName)[0])


# print(studentImgNameList)
# print(studentIDList)


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

# Creating encodings of the "known" faces (i. e., the faces already saved
# in the database)
print("Encoding Started (for known faces)...")
known_EncodingList = findEncodings(studentImgList) # function call
print("Encoding Successful!")
# print(known_EncodingList)


# Save the encodings with their respective IDs in a pickle file
# so that it can be imported in main.py
known_EncodingWithIDsList = [known_EncodingList, studentIDList]

# Creating a file where pickle file is to be dumped
print("Saving the encodings in file...")
file = open("encoding_with_IDs.p", "wb")
pickle.dump(known_EncodingWithIDsList, file)
file.close()
print("File saved successfully!")

# end encoding_generator.py