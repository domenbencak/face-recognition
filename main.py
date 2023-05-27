import threading
import cv2
import os
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

recognized_person = "Unknown"

reference_imgs = {}

# Specify the folder paths where the images are stored
domen_folder_path = "domen"
elon_folder_path = "elon"

# Load reference images for Domen
domen_image_files = [file for file in os.listdir(domen_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
reference_imgs["Domen"] = [cv2.imread(os.path.join(domen_folder_path, file)) for file in domen_image_files]

# Load reference images for Elon
elon_image_files = [file for file in os.listdir(elon_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
reference_imgs["Elon"] = [cv2.imread(os.path.join(elon_folder_path, file)) for file in elon_image_files]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def check_face(frame):
    global recognized_person
    try:
        faces = detect_faces(frame)
        if len(faces) > 0:
            for person, images in reference_imgs.items():
                for reference_img in images:
                    if DeepFace.verify(frame, reference_img.copy())['verified']:
                        recognized_person = person
                        return
        recognized_person = "Unknown"
    except ValueError:
        recognized_person = "Unknown"

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 24 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if recognized_person == "Unknown":
            cv2.putText(frame, recognized_person, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, recognized_person, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
