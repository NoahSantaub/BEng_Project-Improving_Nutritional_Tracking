## https://core-electronics.com.au/guides/raspberry-pi/getting-started-with-yolo-object-and-animal-recognition-on-the-raspberry-pi/
import cv2 as cv
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8
model = YOLO("yolo11m.pt")

testFruitArray=["Image Recognition\\Test Images\\apple.jpg", "Image Recognition\\Test Images\\apple_1.jpg", "Image Recognition\\Test Images\\mango.jpg",
                "Image Recognition\\Test Images\\orange.jpg", "Project_MMME3083\\Code\\Fruit Image DB\\Orange\\Orange0016.png"]
testFruit=testFruitArray[1]

# Load image to be read
imageRaw = cv.imread(testFruit)
#cv.imshow("Image", image)

# Wait for a key press
#cv.waitKey(0)

# Run YOLO model on the captured frame and store the results
results = model(imageRaw)
#print(results[0])

# Output the visual detection data, we will draw this on our camera preview window
annotated_image = results[0].plot()

cv.imshow("Camera", annotated_image)
cv.waitKey(0)


# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open(testFruit)
results = model.predict(source=im1)#, save=True)  # save plotted images
#print(results)

# from ndarray
imagePred = cv.imread(testFruit)
results = model.predict(source=imagePred)#, save=True, save_txt=True)  # save predictions as labels
#print(results)

# from list of PIL/ndarray
#results = model.predict(source=[imageRaw, imagePred])