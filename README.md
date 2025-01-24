### **My Bachelor's Research/Dissertation Project.**
### Close-range photogrammetry and calling databases will estimate the calories of fruit. 
This is the repo for my BEng Individual project where I am exploring the effectiveness of using photogrammetry to calculate food calories compared to other methods on the market.

The point cloud is currently created using Agisoft metashape, but aiming to switch to open source when appropriate. 
A food-recognition model (using TensorFlow) is used to identify the fruit to allow for the calling of the correct data from the sources. 

This project is developed on top of existing work found @ https://www.kaggle.com/code/databeru/classify-15-fruits-with-tensorflow-acc-99-6

Training dataset: https://www.kaggle.com/code/databeru/classify-15-fruits-with-tensorflow-acc-99-6

MUST change/update file locations for every file: where the database is stored, where to save/call the model and image locations

TensorFlow CPU-only
- Tensorflow == 2.18.0 
- Keras == 3.7.0

Tensorflow with Windows native GPU
- Tensorflow == 2.10.1
  - Grapic Drivers -> cuDNN == 8.1 & CUDA == 11.2 https://www.tensorflow.org/install/source#gpu
- Keras == 2.10.0
- Python == 3.9


Libraries used: 
- matplotlib ==3.8.2
- numpy == 1.26.4
- pandas == 2.2.0
- scikit-learn == 1.5.2
- scipy == 1.12.0
- seaborn == 0.13.2
- openCV == 4.11.0.86
