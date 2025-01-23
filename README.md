### **My Bachelor's Research/Dissertation Project.**
### Close-range photogrammetry and calling databases will estimate the calories of fruit. 
This is the repo for my BEng Individual project where I am exploring the effectiveness of using photogrammetry to calculate food calories compared to other methods on the market.

The point cloud is currently created using Agisoft metashape, but aiming to switch to open source when appropriate. 
A food-recognition model (using TensorFlow) is used to identify the fruit to allow for the calling of the correct data from the sources. 

This project is developed on top of existing work found @ https://www.kaggle.com/code/databeru/classify-15-fruits-with-tensorflow-acc-99-6

Training dataset: https://www.kaggle.com/code/databeru/classify-15-fruits-with-tensorflow-acc-99-6

MUST change/update file locations for every file: where the db is stored, where to save/call the model and image locations

CPU-only
- Tensorflow == 2.18.0 
- Keras == 3.7.0

Tensorflow with Windows native GPU, minimum spec Geforce RTX 2060
- Tensorflow == 2.10.1 
- Keras == 2.10.0


Libraries used: 
- matplotlib ==3.8.2
- numpy == 2.0.2
- pandas == 2.2.0
- scikit-learn == 1.52
- scipy == 1.12.0
- seaborn == 0.13.2
