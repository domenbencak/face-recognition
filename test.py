import cv2
import os
from deepface import DeepFace

def save_frame(frame, person):
    folder_path = os.path.join(person.lower(), ".")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    counter = len(os.listdir(folder_path)) + 1
    filename = f"{person}_{counter}.jpg"
    file_path = os.path.join(folder_path, filename)
    cv2.imwrite(file_path, frame)
    print(f"Saved frame for {person} in {file_path}")

def compare_images(frame):
    recognized_person = "Unknown"
    try:
        for person, images in reference_imgs.items():
            for reference_img in images:
                if DeepFace.verify(frame, reference_img.copy())['verified']:
                    recognized_person = person
                    save_frame(frame.copy(), person)
                    break
            if recognized_person != "Unknown":
                break
    except ValueError:
        pass
    return recognized_person

def capture_image():
    _, frame = cap.read()
    cv2.imshow("Capture Image", frame)
    return frame

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

reference_imgs = {}

domen_folder_path = "domen"
elon_folder_path = "elon"

domen_image_files = [file for file in os.listdir(domen_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
reference_imgs["Domen"] = [cv2.imread(os.path.join(domen_folder_path, file)) for file in domen_image_files]

elon_image_files = [file for file in os.listdir(elon_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
reference_imgs["Elon"] = [cv2.imread(os.path.join(elon_folder_path, file)) for file in elon_image_files]

while True:
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("c"):
        captured_frame = capture_image()
        recognized_person = compare_images(captured_frame)
        print(f"Recognized Person: {recognized_person}")

    ret, frame = cap.read()

    if ret:
        cv2.imshow("Camera", frame)

cv2.destroyAllWindows()
cap.release()
