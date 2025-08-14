# BurstTron


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16759183.svg)](https://doi.org/10.5281/zenodo.16759183)



# Detectron2 Image Detection Project

A computer vision project using Detectron2 for object detection and classification of type III and groups of type III SRBs on radio data (Irish-LOFAR images).



## Installation

### Method 1

```bash
# Clone the repository
git clone https://github.com/Hermanlrx/BurstTron.git
cd BurstTron
#This downloads and extracts the data 
curl -L -o /tmp/split_dataset.zip "https://zenodo.org/records/16759183/files/split_dataset.zip?download=1" || { echo "Dataset download failed"; exit 1; }     && unzip /tmp/split_dataset.zip -d BurstTron/     && rm /tmp/split_dataset.zip
```

```bash
# Create a new conda environment (Change based on your preferred env manager)
conda create -n detectron python=3.10
conda activate detectron

# Install the following in order
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install opencv-python
pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Project Structure after downloading data

```
├── BurstTron
|   ├── inference_script.py
|   └── training.py
|   ├── split_dataset/
|     ├── model_final.pth #Usable if user does not want to run training script
│     ├── annotations/
│     │   ├── train.json
│     │   └── val.json
│     ├── train/
│     └── val/
├── TrainingOutput/    #If your are running the training script
│   └── model_final.pth
└── README.md
```

## Usage

### Basic Inference

```bash
python inference.py --input_path /path/to/your/images
```

### Example Commands

```bash
# Process images in a specific directory
python inference.py --input_path "/path/to/your/test_images"

```

### Output

The script will:
1. Process all `.png` files in the input directory
2. Run object detection on each image
3. Generate annotated images with bounding boxes
4. Save results with filename format: `originalname_event.png` or `originalname_no_event_detected.png`
5. Create an `output/` subdirectory in image folder containing all processed images

## Configuration

### Key Parameters for Inference.py

- `SCORE_THRESH_TEST`: Confidence threshold for detections (default: 0.5)
- `MODEL.WEIGHTS`: Path to your trained model weights
- `INPUT_PATH`: Directory containing images to process

### Model Configuration

The script automatically loads:
- Training dataset metadata for class names
- Validation dataset for testing
- Pre-trained model weights from `split_dataset/model_final.pth`

## Dataset Format

This project uses COCO format annotations:

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [
    {"id": 1, "name": "event_type_1"},
    {"id": 2, "name": "event_type_2"}
  ]
}
```




## Model Training

To train your own model:

1. Donwload data from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16314688.svg)](https://doi.org/10.5281/zenodo.16314688)
2. Ensure code is in the same directory as the data files
3. Run training script




## Acknowledgments

- [Detectron2](https://github.com/facebookresearch/detectron2) by Facebook AI Research
- LOFAR telescope data providers
- PyTorch and OpenCV communities


## Support
For issues and questions:
- Open a GitHub issue
