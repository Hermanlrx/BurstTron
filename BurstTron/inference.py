from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
import os
import cv2
import glob
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up argument parser
parser = argparse.ArgumentParser(description="Process images in a specified folder.")
parser.add_argument('--input_path', type=str, default="",
                    help="Path to the input folder containing PNG files")
args = parser.parse_args()



# Use the provided input path
input_path = args.input_path
print(input_path)
output_path = os.path.join(input_path, "output")
os.makedirs(output_path, exist_ok=True)
print(output_path)


train_json = os.path.join(BASE_DIR, "split_dataset/annotations/train.json")
train_images = os.path.join(BASE_DIR, "split_dataset/train")
val_json = os.path.join(BASE_DIR, "split_dataset/annotations/val.json")
val_images = os.path.join(BASE_DIR, "split_dataset/val")

# Register datasets
register_coco_instances("lofar_train", {}, train_json, train_images)
register_coco_instances("lofar_val", {},val_json, val_images)
dataset_dicts = DatasetCatalog.get("lofar_train")
print(f"First sample: {dataset_dicts[0]}")
print(f"Categories: {MetadataCatalog.get('lofar_train').thing_classes}")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("lofar_train",)
cfg.DATASETS.TEST = ("lofar_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.OUTPUT_DIR = os.path.join(BASE_DIR,"Results")


#If attempting to do inference on an example model provided use the following 
#The output should be created in the directory provided 
cfg.MODEL.WEIGHTS = os.path.join(BASE_DIR,"split_dataset/model_final.pth")

#If running the training script and you want to do inference thereafter use the following
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #.5 is a standard number used in these cases 
predictor = DefaultPredictor(cfg)




# Get all PNG files in the directory
file_pattern = os.path.join(input_path, "*.png")

files = glob.glob(file_pattern)

print(files)

for file_path in files:
    
    im = cv2.imread(file_path)
    
    # Run prediction on an image 
    outputs = predictor(im)
    
    # Check if there are any detections (event vs no event)
    instances = outputs["instances"]
    has_event = len(instances) > 0
    event_status = "event" if has_event else "no_event_detected"
    
    # Changing file format to be a bit more uniform 
    filename = os.path.basename(file_path)
    name_part = filename.replace(" ", "_").replace("+00:00", "").replace(".png", "")
    output_filename = f"{name_part}_{event_status}.png"
    
    # Visualize results
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(instances.to("cpu"))
    
    # Save the result
    cv2.imwrite(os.path.join(output_path,output_filename), out.get_image()[:, :, ::-1])
    
    print(f"Processed: {filename} -> {output_filename}")

print("Processing complete!")