from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, hooks
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
import os
import torch
import gc
import cv2
import numpy as np
np.bool = np.bool_ 
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

gc.collect()
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_json = os.path.join(BASE_DIR, "split_dataset/annotations/train.json")
train_images = os.path.join(BASE_DIR, "split_dataset/train")
val_json = os.path.join(BASE_DIR, "split_dataset/annotations/val.json")
val_images = os.path.join(BASE_DIR, "split_dataset/val")

# Register datasets
register_coco_instances("lofar_train", {}, train_json, train_images)
register_coco_instances("lofar_val", {},val_json, val_images)

# Check dataset
dataset_dicts = DatasetCatalog.get("lofar_train")
print(f"First sample: {dataset_dicts[0]}")
print(f"Categories: {MetadataCatalog.get('lofar_train').thing_classes}")

# Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("lofar_train",)
cfg.DATASETS.TEST = ("lofar_val",)
cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
cfg.DATALOADER.NUM_WORKERS = 4
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (2500, 3750)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("lofar_train").thing_classes)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.OUTPUT_DIR = "./TrainingOutput"

# Create output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Custom Trainer with validation
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    
    def run_validation(self):
        if not self.cfg.DATASETS.TEST:
            return
        evaluator = self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])
        data_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        results = inference_on_dataset(self.model, data_loader, evaluator)
        return results

# Create trainer
trainer = CustomTrainer(cfg)

# Training loop with validation
try:
    trainer.resume_or_load(resume=False)
    print("Running initial validation...")
    trainer.run_validation()
    
    print("Starting training...")
    trainer.train()
    
    print("Running final validation...")
    trainer.run_validation()
    
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training failed with error: {e}")
    raise