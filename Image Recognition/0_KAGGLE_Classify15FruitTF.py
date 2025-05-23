# Load the libraries
import tensorflow as tf
print(tf.__version__) #verifies tensorflow version
print(tf.config.list_physical_devices('GPU')) #checks for gpu
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
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from IPython.display import Markdown, display

from time import perf_counter
import pickle

def load_images_from_folder(folder,only_path = False, label = ""):
# Load the paths to the images in a directory or load the images
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
directory = r"C:\Users\nsant\OneDrive\Documents\Uni\Y3\Project_MMME3083\Code\Fruit Image DB"
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
#print(mapper_fruit_names)

df = pd.DataFrame(images, columns = ["fruit", "path"])
test_df = train_test_split(df[['path','fruit']].sample(frac=0.05,random_state=0), test_size=0.2,random_state=0)
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1)

train_df,test_df = train_test_split(df[['path','fruit']].sample(frac=0.05,random_state=0), test_size=0.2,random_state=0)
    # Load the Images with a generator and Data Augmentation
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='fruit',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
    #   rotation_range=30, # Uncomment those lines to use data augmentation
    #   zoom_range=0.15,
    #   width_shift_range=0.2,
    #   height_shift_range=0.2,
    #   shear_range=0.15,
    #   horizontal_flip=True,
    #   fill_mode="nearest"
    )
val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='fruit',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation',
    #   rotation_range=30, # Uncomment those lines to use data augmentation
    #   zoom_range=0.15,
    #   width_shift_range=0.2,
    #   height_shift_range=0.2,
    #   shear_range=0.15,
    #   horizontal_flip=True,
    #   fill_mode="nearest"
    )
test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='fruit',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

#@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
#def custom_fn(x):
#    return x**2
class CustomLayer(keras.layers.Layer):
    def __init__(self, sublayer, **kwargs):
        super().__init__(**kwargs)
        self.sublayer = sublayer

    def call(self, x):
        return self.sublayer(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sublayer": keras.saving.serialize_keras_object(self.sublayer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("sublayer")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)
    

#!############################
#need to find a way to dynamicaly send the file location to
sampleImage1="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project_MMME3083\\Code\\apple.jpg"
bapple="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project_MMME3083\\Code\\apple_1.jpg"
mango="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project_MMME3083\\Code\\mango.jpg"
owang="C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project_MMME3083\\Code\\orange.jpg"

image=[sampleImage1,bapple,mango,owang]
    
for i in range (4):
    img =  plt.imread(image[i])
    img = cv2.resize(img, (150,150))

##!##############################
#model_pkl_file="C:/Users/nsant/OneDrive/Documents/Uni/Y3/Food_Photogrammetry/Code/foodRecognitionClassifier.pkl"
#with open(model_pkl_file, 'rb') as file:  
#    model = pickle.load(file)
# evaluate model 
#y_predict = model.predict(sampleImage)


import joblib
# load model with joblib
#model = joblib.load("C:/Users/nsant/OneDrive/Documents/Uni/Y3/Food_Photogrammetry/Code/foodRecognitionClassifier.sav")
#model.summary()
# evaluate model 
#y_predict = model.predict(sampleImage)


from tensorflow.python.keras.models import load_model
# load model 
model = load_model('C:/Users/nsant/OneDrive/Documents/Uni/Y3/Food_Photogrammetry/Code/foodRecognitionClassifier.keras')
# check model info 
model.summary()
#y_predict = model.predict(sampleImage)

# Pass the custom objects dictionary to a custom object scope and place
# the `keras.models.load_model()` call within the scope.
custom_objects = {"CustomLayer": CustomLayer}
with keras.saving.custom_object_scope(custom_objects):
    model = keras.models.load_model("custom_model.keras")
# Let's check:
np.testing.assert_allclose(model.predict(image[0]))


# Predict the label of the test_images
for x in image:
    predict = model.predict(image)
    print(predict)
#predict = np.argmax(predict,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predict = [labels[k] for k in predict]
# Get the accuracy on the test set
y_test = list(test_df.fruit)
acc = accuracy_score(y_test,predict)
print(acc)
print(f'# Accuracy on the test set: {acc * 100:.2f}%')

# check results
print(classification_report(y_test, predict))
#test
