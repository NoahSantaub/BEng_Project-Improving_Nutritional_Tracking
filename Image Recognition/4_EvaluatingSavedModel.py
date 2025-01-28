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
print(tf.keras.__version__)#version check
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from IPython.display import Markdown, display
from time import perf_counter
#import kagglehub
from time import perf_counter
import pickle

def printmd(string):
    # Print with Markdowns    
    display(Markdown(string))

np.random.seed(0) # Add random seed of training for reproducibility

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

#!Competition of 27 pre-trained architectures - May the best win
# Use only 5% on the pictures to speed up the training
train_df,test_df = train_test_split(df[['path','fruit']].sample(frac=0.05,random_state=0), test_size=0.2,random_state=0)
def create_gen():
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
    
    return train_generator,test_generator,train_images,val_images,test_images

#######################################################################################################################
#! Train architecture with the best result
# Split into train/test datasets using all of the pictures
train_df,test_df = train_test_split(df, test_size=0.1, random_state=0)

# Create the generator
train_generator,test_generator,train_images,val_images,test_images=create_gen()

# Create and train the model
#model = get_model(tf.keras.applications.DenseNet201)
#history = model.fit(train_images, validation_data=val_images, epochs=1, callbacks=[tf.keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup", save_freq='epoch', delete_checkpoint=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)])
model = tf.keras.models.load_model('C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project-Source\\Image Recognition\\Saved Models\\foodRecognitionClassifier.keras')

pd.DataFrame(model.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(model.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()

# Predict the label of the test_images
predict = model.predict(test_images)
predict = np.argmax(predict,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predict = [labels[k] for k in predict]

# Get the accuracy on the test set
y_test = list(test_df.fruit)
acc = accuracy_score(y_test,predict)
print(f'# Accuracy on the test set: {acc * 100:.2f}%')

# Display a confusion matrix
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, predict, normalize='true')
plt.figure(figsize = (10,7))
sns.heatmap(cf_matrix, annot=False, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Display picture of the dataset with their labels
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 12),subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.path.iloc[i]))
    ax.set_title(f"True: {test_df.fruit.iloc[i].split('_')[0]}\nPredicted: {predict[i].split('_')[0]}", fontsize = 15)
plt.tight_layout()
plt.show()
