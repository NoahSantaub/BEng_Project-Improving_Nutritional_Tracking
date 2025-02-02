import cv2 as cv
from PIL import Image
from ultralytics import YOLO
import numpy
testFruitArray=["Image Recognition\\Test Images\\apple.jpg", "Image Recognition\\Test Images\\apple_1.jpg", "Image Recognition\\Test Images\\mango.jpg",
                "Image Recognition\\Test Images\\orange.jpg", "Project_MMME3083\\Code\\Fruit Image DB\\Orange\\Orange0016.png"]
testFruit=testFruitArray[1]

# Load a pretrained YOLO11n-cls Classify model
model = YOLO("yolo11m.pt")

## https://core-electronics.com.au/guides/raspberry-pi/getting-started-with-yolo-object-and-animal-recognition-on-the-raspberry-pi/
imageRaw = cv.imread(testFruit)# Load image to be read
#cv.imshow("Image", image)
#cv.waitKey(0)# Wait for a key press

# Run YOLO model on the captured frame and store the results
results = model(imageRaw)

# Output the visual detection data, we will draw this on our camera preview window
annotated_image = results[0]#.plot()

predictedClass = results[0].boxes.cls.numpy()
predictionConfidance = results[0].boxes.conf.numpy()

probs = results[0].probs  # get classification probabilities
top1_confidence = probs.top1conf  # get confidence of top 1 class
print(f"Top 1 class confidence: {top1_confidence.item():.4f}")


print("Predicted Class: ",predictedClass,"\nPrediction Confidance: ",predictionConfidance)

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
