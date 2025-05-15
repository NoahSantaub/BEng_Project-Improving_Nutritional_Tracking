### **My Bachelor's Research/Dissertation Project.**
### Improving Nutritional Tracking: Combining Close-Range Photogrammetry and Machine Learning. 

This is the repo for my BEng Individual project, where I explore the viability of using photogrammetry to calculate food calories. With the addition of ML to automate the process identifying fruit in the image.

The point cloud is currently created using Agisoft Metashape Professional, but aiming to switch to open source when appropriate. 

MUST change/update file locations for every file: where the database is stored, where to save/call the model and image locations


**Close-Range Photogrammetry**
The phone must be placed at a constant distance away from the target object (fruit/food). Keep the phone at a set location using a phone stand or similar. Have the object placed on a flat object which can be rotated around 360 degrees. (Similar to below with a 3D printed phone stand and notebook. 
![image](https://github.com/user-attachments/assets/77b8f801-4a75-4323-b882-0852f6b50f9a)

Using a matlab script from U. o. Nottingham. "Manufacturing Metrology Team." (https://www.nottingham.ac.uk/research/groups/advanced-manufacturing-technology-research
group/research/manufacturing-metrology-team/index.aspx) (accessed May, 2025). Calibration is conducted via a printed checkerboard being placed at the same location on which the food was/will be placed for image acuisiton. 
![image](https://github.com/user-attachments/assets/ab22b4c8-4bf9-4b8c-ac26-67e3ad69efa4)

The images are uploaded into agisoft (scripting to be created in the future), import calibration file, align photos, reconstruct point cloud with high densification and output with local geometry.

CloudCompare is used to clean up the point cloud and estimate the volume of the food. 

Using Nutritionix API (yet todo), CSV fruit density database (only contains 5 fruits) and volume from above, the predicted calories are calculated. 


**Image Recognition**
The initial concept used image recognition to idenfity the food in the image however it was found to be the wrong approach, and object detection should be used.


A food-recognition model (using TensorFlow) is used to identify the fruit to allow for the calling of the correct data from the sources. 

This project is developed on top of existing work found @ https://www.kaggle.com/code/databeru/classify-15-fruits-with-tensorflow-acc-99-6

Training dataset: https://www.kaggle.com/datasets/chrisfilo/fruit-recognition


**Object Detection**
.....


Written in python 3.9.21
Libraries used:
- matplotlib ==3.8.2
- numpy == 1.26.4
- pandas == 2.2.0
- scikit-learn == 1.5.2
- scipy == 1.12.0
- seaborn == 0.13.2
- openCV == 4.11.0.86

Libraries of Object Detection:
  - ultralytics == 8.3.74
  - torch == 2.6.0+cu126

Libraries required to run Image Recognition files:
  TensorFlow CPU-only
  - Tensorflow == 2.18.0 
  - Keras == 3.7.0
  
  Tensorflow with Windows native GPU
  - Tensorflow == 2.10.1
    - Grapic Drivers -> cuDNN == 8.1 & CUDA == 11.2 https://www.tensorflow.org/install/source#gpu
  - Keras == 2.10.0
  - Python == 3.9
