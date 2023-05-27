import threading
import cv2
import os
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

counter = 0

recognized_person = "Unknown"

reference_imgs = {}

# Specify the folder paths where the images are stored
domen_folder_path = "domen"
elon_folder_path = "elon"

# Get all image files for Domen with extensions .jpg, .jpeg, and .png
domen_image_files = [file for file in os.listdir(domen_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
# Load the images for Domen and add them to the reference_imgs dictionary under the person's name
reference_imgs["Domen"] = [cv2.imread(os.path.join(domen_folder_path, file)) for file in domen_image_files]

# Get all image files for Elon with extensions .jpg, .jpeg, and .png
elon_image_files = [file for file in os.listdir(elon_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
# Load the images for Elon and add them to the reference_imgs dictionary under the person's name
reference_imgs["Elon"] = [cv2.imread(os.path.join(elon_folder_path, file)) for file in elon_image_files]

# Add more persons with their folder paths if needed

def check_face(frame):
    global recognized_person
    try:
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
