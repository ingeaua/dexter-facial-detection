import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt')

image_dir = '../../validare/validare'

detections_dad = []
file_names_dad = []
scores_dad = []

detections_deedee = []
file_names_deedee = []
scores_deedee = []

detections_dexter = []
file_names_dexter = []
scores_dexter = []

detections_mom = []
file_names_mom = []
scores_mom = []

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

        if cls == 0:
            detections_dad.append([xmin, ymin, xmax, ymax])
            scores_dad.append(conf)
            file_names_dad.append(str(image_path.name))
        elif cls == 1:
            detections_deedee.append([xmin, ymin, xmax, ymax])
            scores_deedee.append(conf)
            file_names_deedee.append(str(image_path.name))
        elif cls == 2:
            detections_dexter.append([xmin, ymin, xmax, ymax])
            scores_dexter.append(conf)
            file_names_dexter.append(str(image_path.name))
        else:
            detections_mom.append([xmin, ymin, xmax, ymax])
            scores_mom.append(conf)
            file_names_mom.append(str(image_path.name))


detections_dad = np.array(detections_dad)
file_names_dad = np.array(file_names_dad)
scores_dad = np.array(scores_dad)

detections_deedee = np.array(detections_deedee)
file_names_deedee = np.array(file_names_deedee)
scores_deedee = np.array(scores_deedee)

detections_dexter = np.array(detections_dexter)
file_names_dexter = np.array(file_names_dexter)
scores_dexter = np.array(scores_dexter)

detections_mom = np.array(detections_mom)
file_names_mom = np.array(file_names_mom)
scores_mom = np.array(scores_mom)

os.makedirs('../fisiere_solutie_yolo/task2', exist_ok=True)

np.save('../fisiere_solutie_yolo/task2/detections_dad', detections_dad)
np.save('../fisiere_solutie_yolo/task2/detections_deedee', detections_deedee)
np.save('../fisiere_solutie_yolo/task2/detections_dexter', detections_dexter)
np.save('../fisiere_solutie_yolo/task2/detections_mom', detections_mom)
np.save('../fisiere_solutie_yolo/task2/file_names_dad', file_names_dad)
np.save('../fisiere_solutie_yolo/task2/file_names_deedee', file_names_deedee)
np.save('../fisiere_solutie_yolo/task2/file_names_dexter', file_names_dexter)
np.save('../fisiere_solutie_yolo/task2/file_names_mom', file_names_mom)
np.save('../fisiere_solutie_yolo/task2/scores_dad', scores_dad)
np.save('../fisiere_solutie_yolo/task2/scores_deedee', scores_deedee)
np.save('../fisiere_solutie_yolo/task2/scores_dexter', scores_dexter)
np.save('../fisiere_solutie_yolo/task2/scores_mom', scores_mom)
