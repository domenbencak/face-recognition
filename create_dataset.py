import cv2
import os

def capture_images(num_photos, dataset_name):
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    while count < num_photos:
        ret, frame = cap.read()

        cv2.imshow('Webcam', frame)

        img_name = f'{dataset_name}/photo_{count+1}.jpg'
        cv2.imwrite(img_name, frame)

        count += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_images(250, 'dataset_dejan')