import os
from deepface import DeepFace

def find_most_confident_image(given_picture_path, directory_path):
    distance_scores = {}

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            image_path = os.path.join(root, file)

            distance = calculate_confidence(given_picture_path, image_path)
            distance_scores[file] = distance

    most_confident_image = min(distance_scores, key=distance_scores.get)
    return most_confident_image

def calculate_confidence(img1_path, img2_path):
    result = DeepFace.verify(img1_path, img2_path)
    distance = result['distance']
    print(f"{img2_path} distance is {distance}")
    return distance

given_picture_path = "img_3.png"
directory_path = "images_directory"

most_confident_image = find_most_confident_image(given_picture_path, directory_path)
print(f"The most confident image is: {most_confident_image}")
