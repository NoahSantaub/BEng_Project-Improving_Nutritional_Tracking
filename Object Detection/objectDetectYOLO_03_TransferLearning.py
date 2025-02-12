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

# Load a pretrained model
model = YOLO("yolo11n.pt")
#model.to('cuda')
results = model.train(data="C:/Users/nsant/OneDrive/Documents/Uni/Y3/Project_MMME3083/Code/open-images-v7-COCO-v05\dataset.yaml", epochs=300, imgsz=1280,workers=0, batch=-1, patience=5, optimizer="auto", 
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
                        perspective= 0.0,
                        flipud= 0.0,
                        fliplr= 0.08631,
                        mosaic= 0.42551,
                        mixup= 0.0,
                        copy_paste= 0.0)

#batch=-1 automatically determines the batch size that can be efficiently processed based on your device's capabilities
#patience=5 # training will stop if there's no improvement in validation metrics for 5 consecutive epochs
model.save("yolo11n-TransferLearningV04-LargeFruitDBhyperparam.pt")