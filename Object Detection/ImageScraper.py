import fiftyone as fo
import fiftyone.zoo as foz

# Define the classes you want to include
#fruitList = ["Apple","Banana","Grapefruit","Lemon","Mango","Orange", "Peach", "Pear", "Watermelon", "Tomato", "Strawberry"]
fruitList = ["Apple", "Grape", "Common fig", "Pear", "Strawberry" , "Tomato", "Lemon", "Banana", "Orange", "Peach", "Pear", "Mango", "Pineapple", "Grapefruit", "Pomegranate", "Watermelon", "Cantaloupe"]
# Fruits in Open Images v7 - "Apple", "Grape", "Common fig", "Pear", "Strawberry" , "Tomato", "Lemon", "Banana, "Orange", "Peach", "Pear", "Mango", "Pineapple", "Grapefruit", "Pomegranate", "Watermelon", "Cantaloupe"
# Veg in Open Images v7 - "Artichoke", "Asparagus", "Bell pepper", "Broccoli", "Cabbage", "Carrot", "Cucumber", "Mushroom", "Potato", "Pumpkin", "Radish", "Salad", "Winter melon", "Zuccini"


# Images that contain all `label_types` and `classes` will be
# prioritized first, followed by images that contain at least one of
# the required `classes`. If there are not enough images matching
# `classes` in the split to meet `max_samples`, only the available
# images will be loaded.
# Detections - 
# Classifications - 
# Segmentations - 

# The splits to export
splits = ["train", "test", "validation"]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    #split=splits, #"train", # specify splits to load, ('train', 'test', 'validation'); is none provided all splits are loaded 
    label_types=["detections"], # yolo only supports detections label # label types to load - ("detections", "classifications", "relationships", "points", segmentations") - defult all labels loaded
    classes=fruitList,
    max_samples=17000, #estimating 1000 images per class # maximum number of samples to load per split
    seed=0,# ensures repeatability
)

export_dir="C:/Users/nsant/OneDrive/Documents/Uni/Y3/Project_MMME3083/Code/open-images-v7-COCO-v04" #export_dir = "/path/for/yolov5-dataset"
label_field = "ground_truth"  # for example

# The dataset or view to export
# We assume the dataset uses sample tags to encode the splits to export
#dataset_or_view = fo.load_dataset(dataset)

# Export the splits
for split in splits:
    split_view = dataset.match_tags(split)
    split_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        #label_field=label_field,
        split=split,
        classes=fruitList,# All splits must use the same classes list
    )

#session = fo.launch_app(dataset.view(), port=5151)