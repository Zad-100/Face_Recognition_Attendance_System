import firebase_admin
from firebase_admin import credentials, db
import json

cred = credentials.Certificate("facial-recog-attendance-firebase-adminsdk-m0o07-0b6db63d1f.json")
firebase_admin.initialize_app(cred, {
    "databaseURL" : "https://facial-recog-attendance-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref=db.reference("Students")

with open("students_data.json", "r") as f:
	file_contents = json.load(f)

ref.set(file_contents)

# end add_data_to_firebase.py