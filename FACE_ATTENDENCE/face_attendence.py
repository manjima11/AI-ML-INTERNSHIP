import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image

# ---------------------------
# CREATE REQUIRED FOLDERS
# ---------------------------
os.makedirs("dataset", exist_ok=True)
os.makedirs("trainer", exist_ok=True)

# ---------------------------
# TRAIN MODEL
# ---------------------------
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []

    for file in os.listdir("dataset"):
        path = os.path.join("dataset", file)

        parts = file.split(".")
        if len(parts) < 3:
            continue

        user_id = int(parts[1])

        img = Image.open(path).convert('L')
        img_numpy = np.array(img, 'uint8')

        faces.append(img_numpy)
        ids.append(user_id)

    if len(faces) == 0:
        return False

    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer/trainer.yml")

    return True
# ---------------------------
# REGISTER FACE
# ---------------------------
def register_face(name, user_id):
    if not name or not user_id.isdigit():
        st.error("Enter valid Name and Numeric ID")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    sample_count = 0

    st.info("Capturing face samples... Look at the camera")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_count += 1
            cv2.imwrite(
                f"dataset/{name}.{user_id}.{sample_count}.jpg",
                gray[y:y+h, x:x+w]
            )
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Register Face", img)

        if cv2.waitKey(1) == 27 or sample_count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()

    trained = train_model()

    if trained:
        st.success("Face Registered & Model Trained Successfully!")
    else:
        st.error("Training Failed!")


# ---------------------------
# MARK ATTENDANCE
# ---------------------------
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Prevent duplicate attendance on same day
    if ((df['Name'] == name) & (df['Date'] == date)).any():
        return False

    new_entry = {"Name": name, "Date": date, "Time": time}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv("attendance.csv", index=False)
    return True


# ---------------------------
# RECOGNIZE FACE (ONE TIME CAPTURE)
# ---------------------------
def recognize_face():
    if not os.path.exists("trainer/trainer.yml"):
        st.error("Model not trained yet. Register face first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)

    # ID → Name mapping
    names = {}
    for file in os.listdir("dataset"):
        parts = file.split(".")
        if len(parts) >= 3:
            user_id = int(parts[1])
            names[user_id] = parts[0]

    st.info("Camera started... Look at the camera")

    recognized = False

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            user_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                name = names.get(user_id, "Unknown")
                attendance_added = mark_attendance(name)
                recognized = True

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, f"{name} - Marked",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

                cv2.imshow("Attendance Camera", img)
                cv2.waitKey(2000)
                break
            else:
                cv2.putText(img, "Unknown",
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

        cv2.imshow("Attendance Camera", img)

        if recognized:
            break

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    if recognized:
        st.success("Attendance Marked Successfully!")
    else:
        st.warning("Face Not Recognized")


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Face Detection Based Attendance System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Register Face", "Mark Attendance", "View Attendance"]
)

if menu == "Register Face":
    st.subheader("Register New Student")
    name = st.text_input("Enter Name")
    user_id = st.text_input("Enter Numeric ID")

    if st.button("Start Registration"):
        register_face(name, user_id)

elif menu == "Mark Attendance":
    st.subheader("Start Attendance")
    if st.button("Start Camera"):
        recognize_face()

elif menu == "View Attendance":
    st.subheader("Attendance Report")

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.warning("No attendance records found.")