import fiftyone as fo
import fiftyone.zoo as foz

# Define the classes you want to include
fruitList = ["Apple","Banana","Mango","Orange", "Pear", "Peach",]
#fruitList = ["Apple","Banana","Grapefruit","Lemon","Mango","Orange", "Pear", "Peach", , "Watermelon", "Tomato", "Strawberry"]
# 
# Images that contain all `label_types` and `classes` will be
# prioritized first, followed by images that contain at least one of
# the required `classes`. If there are not enough images matching
# `classes` in the split to meet `max_samples`, only the available
# images will be loaded.
# Detections - 
# Classifications - 
# Segmentations - 
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    #split="train", # specify splits to load; is none provided all splits are loaded 
    label_types=["detections","classifications"], # label types to load - ("detections", "classifications", "relationships", "points", segmentations") - defult all labels loaded
    classes=fruitList,
    max_samples=1000,  # manimum number of samples to load per split
    seed=0,# ensures repeatability
)

# Export the dataset to COCO format
dataset.export(
    export_dir="C:/Users/nsant/OneDrive/Documents/Uni/Y3/Project_MMME3083/Code/open-images-v7-COCO",
    dataset_type=fo.types.COCODetectionDataset, # export the data into the coco format
)
session = fo.launch_app(dataset.view(), port=5151)