import threading
import cv2
import os
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
was_saved = False

recognized_person = "Unknown"

reference_imgs = {}

domen_folder_path = "domen"
elon_folder_path = "elon"

domen_image_files = [file for file in os.listdir(domen_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
reference_imgs["Domen"] = [cv2.imread(os.path.join(domen_folder_path, file)) for file in domen_image_files]

elon_image_files = [file for file in os.listdir(elon_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
reference_imgs["Elon"] = [cv2.imread(os.path.join(elon_folder_path, file)) for file in elon_image_files]


# TODO - dodaj Aljaza in Dejana

def save_frame(frame, person):
    global was_saved
    if not was_saved:
        folder_path = os.path.join(person.lower(), ".")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        counter_name = len(os.listdir(folder_path)) + 1
        filename = f"{person}_{counter_name}.jpg"
        file_path = os.path.join(folder_path, filename)
        cv2.imwrite(file_path, frame)
        print(f"Saved frame for {person} in {file_path}")
        was_saved = True
    else:
        pass
        # print("Already saved")


def check_face(frame):
    global recognized_person
    try:
        for person, images in reference_imgs.items():
            for reference_img in images:
                if DeepFace.verify(frame, reference_img.copy())['verified']:
                    recognized_person = person
                    threading.Thread(target=save_frame, args=(frame.copy(), person)).start()
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
            if recognized_person == "Domen":
                cv2.putText(frame, recognized_person, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif recognized_person == "Elon":
                cv2.putText(frame, recognized_person, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
