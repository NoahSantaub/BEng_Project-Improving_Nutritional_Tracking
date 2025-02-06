import fiftyone as fo
import fiftyone.zoo as foz

# Define the classes you want to include
classes = ["Apple","Grapefruit","Lemon","Mango","Orange", "Pear", "Peach", "Pineapple", "Watermelon", "Tomato", "Strawberry"]

# Load only point labels (potentially negative or mixed) for 25 samples
# from the validation split for tortoise and sea turtle classes
#
# Images that contain all `label_types` and `classes` will be
# prioritized first, followed by images that contain at least one of
# the required `classes`. If there are not enough images matching
# `classes` in the split to meet `max_samples`, only the available
# images will be loaded.
#
# Images will only be downloaded if necessary
# Load the Open Images V7 dataset
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections", "segmentations", "points"],
    classes=classes,
    max_samples=1000,  # Adjust this number based on your needs
    dataset_dir="/content/open-images-v7"
)

# Export the dataset to YOLO format
dataset.export(
    export_dir="/content/open-images-v7-yolo",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    classes=classes,
    split="train"
)
session = fo.launch_app(dataset)
