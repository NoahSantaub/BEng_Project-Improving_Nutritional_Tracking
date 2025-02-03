import cv2 as cv
import numpy
from ultralytics import YOLO
# Docs & Resources used 
# https://docs.ultralytics.com/modes/predict/ # https://docs.ultralytics.com/reference/engine/results/
# https://core-electronics.com.au/guides/raspberry-pi/getting-started-with-yolo-object-and-animal-recognition-on-the-raspberry-pi/

testFruitArray=["Image Recognition\\Test Images\\apple.jpg", "Image Recognition\\Test Images\\apple_1.jpg", "Image Recognition\\Test Images\\mango.jpg",
                "Image Recognition\\Test Images\\orange.jpg", "Project_MMME3083\\Code\\Fruit Image DB\\Orange\\Orange0016.png"]
testFruit=testFruitArray[1]

# Load a pretrained YOLO11n-cls Classify model
model = YOLO("yolo11m.pt")

imageRaw = cv.imread(testFruit)# Load image to be read
#cv.imshow("Image", image)
#cv.waitKey(0)# Wait for a key press

# Run YOLO model on the captured frame and store the results
results = model(imageRaw)

# Output the visual detection data, we will draw this on our camera preview window
annotated_image = results[0].plot()
cv.imshow("Camera", annotated_image)
cv.waitKey(0)

predictedClass = results[0].boxes.cls.numpy()
predictionConfidance = results[0].boxes.conf.numpy()

strippedClassID = str(predictedClass).replace('[','').replace(']','').replace(' ', '')
className=model.names.get(int(strippedClassID))
strippedClassConfidance = str(predictionConfidance).replace('[','').replace(']','').replace(' ', '')

print("Predicted Class: ",className,
    "\nPrediction Confidance: ",strippedClassConfidance)


boxes = results[0].boxes  # Boxes object for bounding box outputs
masks = results[0].masks  # Masks object for segmentation masks outputs
keypoints = results[0].keypoints  # Keypoints object for pose outputsprint
probs = results[0].probs  # Probs object for classification outputs
obb = results[0].obb  # Oriented boxes object for OBB outputs

"""
# Run inference on an image
results = model(testFruit)  # results list
for result in results:
    im = result.plot()
    #im.show()
value=results.plot(conf=True)
#value=result.boxes.cuda().numpy()
results.show()
print(value)
"""
