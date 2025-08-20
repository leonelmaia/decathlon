# Case Study: Video Analytics & Turnover Forecasting

This repository contains two independent projects:

Video Analytics → Detecting and counting customers in store footage.

Turnover Forecasting → Predicting weekly store-department turnover using historical data.

Both projects are documented and reproducible.

## Setup Instructions

We recommend using Python 3.10+ and a virtual environment.

1. Create and activate a virtual environment
```
# Create a venv
python -m venv venv  

# Activate it
source venv/bin/activate 
 ```

 2. Install requirements
 ```
pip install -r requirements.txt
 ```


## Part 1: Video Analytics

Detects customers visible in store footage.

Uses YOLOv11 for detection and two tracking approaches: ByteTrack and DeepSORT.

Outputs an annotated video with bounding boxes, track IDs, and counts.

### How to run

Once the virtual environment is active, run:
 ```
python video_analysis.py --input video.mp4 --output out.mp4 --model yolo11x.pt --conf 0.1 --iou 0.5
 ```


## Part 2: Turnover Forecasting

Forecasts the next 8 weeks of turnover at the store-department level.

Uses historical sales (train.csv.gz) and store features (bu_feat.csv.gz).

Provides a full pipeline: exploration → preprocessing → modeling → forecasting.

How to run

First, make the virtual environment available as a Jupyter kernel:
 ```
pip install ipykernel  
python -m ipykernel install --user --name=turnover-env --display-name "Python (turnover-env)"
 ```

Open Jupyter Notebook:
 ```
jupyter notebook

 ```

Select the kernel Python (turnover-env).

Run the two notebooks:

01_exploration.ipynb → preliminary exploration and insights.

02_forecasting.ipynb → end-to-end pipeline with final predictions.

## Notes

Video Analytics was tested in Google Colab (GPU T4) for speed.

Turnover Forecasting was developed and run locally.

Each part has its own README with detailed methodology, challenges, and results.