import cv2 as cv
import numpy
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
if torch.cuda.is_available():
    print("gpu detected")
else:
    print("gpu not detected!")
    exit()
# Docs & Resources used 
# https://docs.ultralytics.com/modes/predict/ # https://docs.ultralytics.com/reference/engine/results/
# https://core-electronics.com.au/guides/raspberry-pi/getting-started-with-yolo-object-and-animal-recognition-on-the-raspberry-pi/

testFruitArray=["Image Recognition\\Test Images\\apple.jpg", "Image Recognition\\Test Images\\apple_1.jpg", "Image Recognition\\Test Images\\FruitBowl.jpg",
                "Image Recognition\\Test Images\\orange.jpg", "Project_MMME3083\\Code\\Fruit Image DB\\Orange\\Orange0016.png", 
                "Image Recognition\\Test Images\\Multi_1.jpg", "Image Recognition\\Test Images\\Multi_2.jpg", "Image Recognition\\Test Images\\Multi_3.jpg", 
                "Image Recognition\\Test Images\\Multi_4.jpg", "Image Recognition\\Test Images\\Multi_5.jpg"]
testFruit=testFruitArray[7]

# Load a pretrained model
model = YOLO("yolo11n.pt")
#model.to('cuda')
results = model.train(data="C:/Users/nsant/OneDrive/Documents/Uni/Y3/Project_MMME3083/Code/open-images-v7-COCO-v04\dataset.yaml", epochs=300, imgsz=640,workers=0, batch=-1, patience=5, optimizer="auto", 
lr0= 0.00269,
lrf= 0.00288,
momentum= 0.73375,
weight_decay= 0.00015,
warmup_epochs= 1.22935,
warmup_momentum= 0.1525,
box= 18.27875,
cls= 1.32899,
dfl= 0.56016,
hsv_h= 0.01148,
hsv_s= 0.53554,
hsv_v= 0.13636,
degrees= 0.0,
translate= 0.12431,
scale= 0.07643,
shear= 0.0,
perspective= 0.0,
flipud= 0.0,
fliplr= 0.08631,
mosaic= 0.42551,
mixup= 0.0,
copy_paste= 0.0)
#batch=-1 automatically determines the batch size that can be efficiently processed based on your device's capabilities
#patience=5 # training will stop if there's no improvement in validation metrics for 5 consecutive epochs
model.save("yolo11n-TransferLearningV04-OptimisedHyperparameters.pt")

imageRaw = cv.imread(testFruit)# Load image to be read
#cv.imshow("Image", imageRaw), cv.waitKey(0)# output original image # Wait for a key press

# Run YOLO model on the captured frame and store the results
results = model(imageRaw) # dependant on the number of images provided, imageRaw == index[0]
#results = model.predict(source=imageRaw, conf=0.5)

# Output the visual detection data, we will draw this on our camera preview window
annotated_image = results[0].plot()
#cv.imshow("Annotated Image", annotated_image), cv.waitKey(0)

imageDisplay=numpy.concatenate((imageRaw, annotated_image), axis=1)
imageDisplay=cv.resize(imageDisplay,(1600,600))
cv.imshow("Image Comparison", imageDisplay), cv.waitKey(0) # output original image with annotated horizontal

predictedClass = results[0].boxes.cls.numpy()
predictionConfidance = results[0].boxes.conf.numpy()

for index in range(predictedClass.size):
    className=model.names.get(int(predictedClass[index]))
    strippedClassConfidance = str(predictionConfidance[index]).replace('[','').replace(']','').replace(' ', '') # strips & sanatises text
    print("Item No. ", index, "\nPredicted Class: ",className,"\nPrediction Confidance: ",strippedClassConfidance)