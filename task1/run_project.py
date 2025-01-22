import os
import numpy as np
import torch
import glob
import cv2 as cv
import ntpath
import timeit
from skimage.feature import hog
from neural_network import load_model
from helpers import non_maximal_suppression
# from feature_extraction import *


def run(test_images_path, dim_hog_cell = 6, dim_window = 36, score_threshold = 0.0, scale_factor=0.8):

    model = load_model('model.pth', input_length=1764)

    test_images_files = test_images_path + '/*.jpg'
    test_files = glob.glob(test_images_files)
    detections = None
    scores = np.array([])
    file_names = np.array([])

    num_test_images = len(test_files)

    for i in range(num_test_images):

        start_time = timeit.default_timer()
        img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
        original_img = img.copy()

        image_scores = []
        image_detections = []
        short_name = ntpath.basename(test_files[i])

        current_scale = 1.0

        while img.shape[0] >= dim_window and img.shape[1] >= dim_window:

            hog_descriptors = hog(img, pixels_per_cell=(dim_hog_cell, dim_hog_cell),
                                  cells_per_block=(2, 2), feature_vector=False)

            num_cols = img.shape[1] // dim_hog_cell - 1
            num_rows = img.shape[0] // dim_hog_cell - 1
            num_cell_in_template = dim_window // dim_hog_cell - 1

            for y in range(0, num_rows - num_cell_in_template):
                for x in range(0, num_cols - num_cell_in_template):
                    descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                    score = model.get_score(torch.from_numpy(descr).float())
                    if score > score_threshold:
                        x_min = int(x * dim_hog_cell * (1 / current_scale))
                        y_min = int(y * dim_hog_cell * (1 / current_scale))
                        x_max = int((x * dim_hog_cell + dim_window) * (1 / current_scale))
                        y_max = int((y * dim_hog_cell + dim_window) * (1 / current_scale))
                        image_detections.append([x_min, y_min, x_max, y_max])
                        image_scores.append(score)

            img = cv.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
            current_scale *= scale_factor

        if len(image_scores) > 0:
            image_detections, image_scores = non_maximal_suppression(np.array(image_detections),
                                                                          np.array(image_scores), original_img.shape)
        if len(image_scores) > 0:
            if detections is None:
                detections = image_detections
            else:
                detections = np.concatenate((detections, image_detections))
            scores = np.append(scores, image_scores)
            image_names = [short_name for _ in range(len(image_scores))]
            file_names = np.append(file_names, image_names)

        end_time = timeit.default_timer()
        print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
              % (i+1, num_test_images, end_time - start_time))

    return detections, scores, file_names


# ------------------------------ parametrii ------------------------------

# negative_desc_file = 'negative_desc-bun.npy'
# positive_desc_file = 'positive_desc.npy'

score_threshold = 0
scale_factor = 0.98

dim_window = 64
dim_hog_cell = 8

# neg_per_image = 100
# min_patch_size = 10
# max_patch_size = 210
# max_iou = 0.35

# batch_size = 128
# num_epochs = 18

test_images_path = '../../validare/validare'

# ------------------------------ run ------------------------------

# pos_desc = get_positive_hog(dim_window=dim_window, dim_hog_cell=dim_hog_cell, augument=True)
# neg_desc = get_negative_hog(dim_window=dim_window, dim_hog_cell=dim_hog_cell, neg_per_image=neg_per_image, min_patch_size=min_patch_size, max_patch_size=max_patch_size, max_iou=max_iou)
# np.save(positive_desc_file, pos_desc)
# np.save(negative_desc_file, neg_desc)

# model = get_model(negative_desc_file, positive_desc_file, batch_size=batch_size, num_epochs=num_epochs)
# save_model(model, 'model.pth')

os.makedirs('../fisiere_solutie/task1', exist_ok=True)

detections, scores, file_names = run(test_images_path=test_images_path, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)

np.save('../fisiere_solutie/task1/detections_all_faces', detections)
np.save('../fisiere_solutie/task1/file_names_all_faces', file_names)
np.save('../fisiere_solutie/task1/scores_all_faces', scores)
