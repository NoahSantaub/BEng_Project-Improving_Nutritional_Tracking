# Load the libraries
import tensorflow as tf
print("TensorFlow Version: ", tf.__version__) #verifies tensorflow version
print("Check for GPU allocation: ",tf.config.list_physical_devices('GPU')) #checks for gpu
import pandas as pd
import numpy as np
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import keras
print("keras Version: ",tf.keras.__version__)#version check
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from IPython.display import Markdown, display
from time import perf_counter
#import kagglehub
startTime=time.perf_counter()

def load_images_from_folder(folder,only_path = False, label = ""):
# Load the paths to the images in a directory
# or load the images
    if only_path == False:
        images = []
        for filename in os.listdir(folder):
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images
    else:
        path = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder,filename)
            if img_path is not None:
                path.append([label,img_path])
        return path

# Load the paths on the images
images = []
directory = r"C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project_MMME3083\\Code\\Fruit Image DB"
for f in os.listdir(directory):
    if "png" in os.listdir(directory+'\\'+f)[0]:
        images += load_images_from_folder(directory+'\\'+f,True,label = f)
    else: 
        for d in os.listdir(directory+'\\'+f):
            images += load_images_from_folder(directory+"\\"+f+"\\"+d,True,label = f)
            
# Create a dataframe with the paths and the label for each fruit
df = pd.DataFrame(images, columns = ["fruit", "path"])

# Shuffle the dataset
from sklearn.utils import shuffle
df = shuffle(df, random_state = 0)
df = df.reset_index(drop=True)

# Assign to each fruit a specific number
fruit_names = sorted(df.fruit.unique())
mapper_fruit_names = dict(zip(fruit_names, [t for t in range(len(fruit_names))]))
df["label"] = df["fruit"].map(mapper_fruit_names)
#{'Apple': 0, 'Banana': 1, 'Carambola': 2, 'Guava': 3, 'Kiwi': 4, 'Mango': 5, 'Orange': 6, 'Peach': 7, 'Pear': 8, 'Persimmon': 9, 'Pitaya': 10, 'Plum': 11, 'Pomegranate': 12, 'Tomatoes': 13, 'muskmelon': 14}
#print(mapper_fruit_names)
#exit()

##* Loading images to be tested ##
#need to find a way to dynamicaly send the file location to
apple="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Test Images\\apple.jpg"
apple_1="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Test Images\\apple_1.jpg"
mango="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Test Images\\mango.jpg"
orange="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Test Images\\orange.jpg"
orangeTrainedWith="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project_MMME3083\\Code\\Fruit Image DB\\Orange\\Orange0016.png"

testFruitArray=[apple,apple_1,mango,orange,orangeTrainedWith]
testFruit=testFruitArray[2]
print(testFruit)
unknownFruit = plt.imread(testFruit)
plt.imshow(unknownFruit)
plt.show()
unknownFruit=(cv2.resize(unknownFruit, (224,224)))
plt.imshow(unknownFruit)
plt.show()
unknownFruit=unknownFruit.reshape(1,224,224,3)

##* Loading Pre-trained Saved Model to quickly identify the fruit##


loadingAModel = tf.keras.models.load_model('C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Saved Models\\foodRecognitionClassifier-MobileNetV2.keras')
#loadingAModel = tf.keras.models.load_model('C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Saved Models\\foodRecognitionClassifier_DenseNet201.keras')
#loadingAModel = tf.keras.models.load_model('C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Saved Models\\foodRecognitionClassifier.keras')
#loadingAModel.summary()

# Predict the label of the test_images
predict = loadingAModel.predict(unknownFruit)
#print(predict)
predictedClass = np.argmax(predict,axis=1)
print("Predicted Class: ",predictedClass)

#value=df.loc['fruit'].value[13]
#value=df.at(13, 'fruit')
strippedText = str(predictedClass).replace('[','').replace(']','')
#print(strippedText)
#value=test_df['fruit'].values[int(strippedText)]
value=list(mapper_fruit_names.keys()) [list(mapper_fruit_names.values()).index(int(strippedText))]
print("Predicted Value: ", value)

confidence_score = tf.math.reduce_max(predict, axis=1) 
print("Confidence score: ", confidence_score)

endTime=time.perf_counter()
elapsedTimeToPredict=endTime-startTime
print(f"Time taken: {elapsedTimeToPredict:.6f} seconds")
