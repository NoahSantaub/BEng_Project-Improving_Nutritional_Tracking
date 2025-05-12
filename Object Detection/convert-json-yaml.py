## Converts the json file into yaml to be read into the YOLO model for transfer learning
import yaml
import json
with open('C:/Users/nsant/OneDrive/Documents/Uni/Y3/Project_MMME3083/Code/open-images-v7-COCO/labels.json', 'r') as file:
    configuration = json.load(file)
with open('C:/Users/nsant/OneDrive/Documents/Uni/Y3/Project_MMME3083/Code/open-images-v7-COCO/openImagesV7-SubsetFruit-V001.yaml', 'w') as yaml_file:
    yaml.dump(configuration, yaml_file)