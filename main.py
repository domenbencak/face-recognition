import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

recognized_person = "Unknown"

reference_imgs = []
reference_imgs.append(cv2.imread("domen/domen5.jpg"))  # Add reference image 1
reference_imgs.append(cv2.imread("elon/elon.jpg"))  # Add reference image 2


# Add more reference images if needed

def check_face(frame):
    global recognized_person
    try:
        for i, reference_img in enumerate(reference_imgs):
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                if i == 0:
                    recognized_person = "Domen"
                elif i == 1:
                    recognized_person = "Elon"
                break
        else:
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
        elif recognized_person == "Domen":
            cv2.putText(frame, recognized_person, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, recognized_person, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
