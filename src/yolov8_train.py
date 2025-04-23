from ultralytics import YOLO
from IPython.display import Image
import cv2
import time
import os

model = YOLO('yolov8n.pt')
datapath=os.path.abspath("../datasets/ctw_and_textocr/data.yaml")
results = model.train(data=datapath, epochs=100, imgsz=640, device=0, batch=10)
