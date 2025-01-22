import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt')

image_dir = '../../validare/validare'

detections = []
file_names = []
scores = []

image_files = [f for f in Path(image_dir).glob('*') if f.suffix == '.jpg']

for image_path in image_files:
    img = Image.open(image_path)
    results = model(img)

    boxes = results.xywh[0].cpu().numpy()

    for box in boxes:
        x_center, y_center, width, height, conf, cls = box

        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)

        detections.append([xmin, ymin, xmax, ymax])
        scores.append(conf)
        file_names.append(str(image_path.name))

detections = np.array(detections)
scores = np.array(scores)
file_names = np.array(file_names)

os.makedirs('../fisiere_solutie_yolo/task1', exist_ok=True)


np.save('../fisiere_solutie_yolo/task1/detections_all_faces', detections)
np.save('../fisiere_solutie_yolo/task1/file_names_all_faces', file_names)
np.save('../fisiere_solutie_yolo/task1/scores_all_faces', scores)
