import os
import numpy as np
from PIL import Image
from keras.utils import np_utils

dataset_path = './dataset_dejan/'
image_size = (224, 224)
class_labels = ['class1', 'class2', 'class3']

class_to_label = {class_label: idx for idx, class_label in enumerate(class_labels)}

images = []
labels = []

for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img)
    images.append(img_array)

    class_label = image_name.split('.')[0]  
    label = class_to_label.get(class_label, -1)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

num_classes = len(class_labels)
labels = np_utils.to_categorical(labels, num_classes)

for label, image in zip(labels, images):
    print("Label:", label)
    print("Image shape:", image.shape)
    print("----")
