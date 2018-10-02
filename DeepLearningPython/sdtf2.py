
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from tqdm import tqdm

DATADIR = "/home/sasank/Documents/Datasets/PetImages" #location of pet images dataset

CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 50

training_data = []

def create_training_data():

    for category in CATEGORIES: #dogs, then cats

        path = os.path.join(DATADIR, category) #create path to dogs and cats (DATADIR)
        class_num = CATEGORIES.index(category) #get the classification (0 = DOG or 1 = CAT)

        for img in tqdm(os.listdir(path)): #for each image
            try:
                #convert to array
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                #resize to normalized size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e: #bad outputs
                pass

create_training_data()

print(len(training_data))

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle
#saving the model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
