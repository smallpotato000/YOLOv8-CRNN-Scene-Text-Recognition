from ultralytics import YOLO
from IPython.display import Image
import cv2
import time
import os
import sys

def process_one_image(input_image_path):
    model = YOLO('runs/detect/train/weights/best.pt')
    img = cv2.imread(input_image_path)
    results = model(img)
    output_image_dir=os.path.splitext(input_image_path)[0] + "_out"
    result=results[0]
    if(result.boxes == None):
        print("No result.")
    else:
        n = 0
        for box in result.boxes:
            x1f,y1f,x2f,y2f=box.xyxy[0]
            x1=int(x1f)
            y1=int(y1f)
            x2=int(x2f)+1
            y2=int(y2f)+1
            crop_img = img[y1:y2, x1:x2]
            os.makedirs(output_image_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_image_dir,f"part_{n}.png"), crop_img)
            n=n+1
            print("Results saved to: " + output_image_dir)

def print_usage():
    print("python yolov8_predict.py -f image_path")

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print_usage()
    if(sys.argv[1] == "-f"):
        input_image_path = sys.argv[2]
        process_one_image(input_image_path)

    else:
        print_usage()
