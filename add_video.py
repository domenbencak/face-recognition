import cv2
import os
import time
from deepface import DeepFace

def save_frame(frame, person, directory):
    folder_path = os.path.join(directory, person.lower())
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    counter = len(os.listdir(folder_path)) + 1
    filename = f"{person.capitalize()}_{counter}.jpg"  # Save with capitalized person's name
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
                    save_frame(frame, person, directory)  # Save the recognized frame
                    break
            if recognized_person != "Unknown":
                break
    except ValueError:
        pass
    return recognized_person

def capture_image(duration):
    frames = []
    start_time = time.time()
    end_time = start_time + duration
    while time.time() < end_time:
        _, frame = cap.read()
        frames.append(frame)
        cv2.imshow("Capture Image", frame)
        cv2.waitKey(500)  # Capture every 0.5 seconds
    return frames

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

directory = "images"  # Directory where images will be saved

# Initialize reference images
reference_imgs = {}
for directory_name in os.listdir(directory):
    folder_path = os.path.join(directory, directory_name)
    if os.path.isdir(folder_path):
        image_files = [file for file in os.listdir(folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
        reference_imgs[directory_name.capitalize()] = [cv2.imread(os.path.join(folder_path, file)) for file in image_files]

while True:
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("a"):
        capture_duration = 5  # Capture images for 5 seconds
        print(f"Capturing images for {capture_duration} seconds...")
        captured_frames = capture_image(capture_duration)
        person_name = input("Enter your name: ")
        if person_name:
            for frame in captured_frames:
                save_frame(frame, person_name, directory)
            # Reload reference images after adding new images
            reference_imgs = {}
            for directory_name in os.listdir(directory):
                folder_path = os.path.join(directory, directory_name)
                if os.path.isdir(folder_path):
                    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
                    reference_imgs[directory_name.capitalize()] = [cv2.imread(os.path.join(folder_path, file)) for file in image_files]
        else:
            print("Invalid name. Images not saved.")

    if key == ord("c"):
        _, frame = cap.read()
        recognized_person = compare_images(frame)
        print(f"Recognized Person: {recognized_person}")

    ret, frame = cap.read()

    if ret:
        cv2.imshow("Camera", frame)

cv2.destroyAllWindows()
cap.release()
