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
                "Image Recognition\\Test Images\\Multi_4.jpg", "Image Recognition\\Test Images\\Multi_5.jpg", "Image Recognition\\Test Images\\TestSubjects1.jpg",
                "Image Recognition\\Test Images\\testImages.jpg"]
testFruit=testFruitArray[10]

# Load custom trained model
#model = YOLO("yolo11m.pt")
model = YOLO("yolo11n.pt")
#model = YOLO("runs/detect/train3/weights/best.pt")

imageRaw = cv.imread(testFruit)# Load image to be read
#cv.imshow("Image", imageRaw), cv.waitKey(0)# output original image # Wait for a key press

# Run YOLO model on the captured frame and store the results
results = model.predict(imageRaw, device="cpu", conf=0.8) # dependant on the number of images provided, imageRaw == index[0]
#results = model.predict(source=imageRaw, conf=0.5)
annotated_image = results[0].plot()
#cv.imshow("Annotated Image", annotated_image), cv.waitKey(0)

imageDisplay=numpy.concatenate((imageRaw, annotated_image), axis=1)
imageDisplay=cv.resize(imageDisplay,(1600,600))
cv.imshow("Image Comparison", imageDisplay)
cv.waitKey(0) # output original image with annotated horizontal
predictedClass = results[0].boxes.cls.numpy()
predictionConfidance = results[0].boxes.conf.numpy()

for index in range(predictedClass.size):
    className=model.names.get(int(predictedClass[index]))
    strippedClassConfidance = str(predictionConfidance[index]).replace('[','').replace(']','').replace(' ', '') # strips & sanatises text
    print("Item No. ", index, "\nPredicted Class: ",className,"\nPrediction Confidance: ",strippedClassConfidance)