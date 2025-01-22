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


def run(test_images_path, model, dim_hog_cell=6, dim_window=36, score_threshold=0.0, scale_factor=0.8):

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

# negative_desc_file = 'negative_desc.npy'
# positive_desc_file = 'positive_desc.npy'

score_threshold = 0
scale_factor = 0.98

dim_window = 64
dim_hog_cell = 8

# neg_per_image = 115
# min_patch_size = 10
# max_patch_size = 210
# max_iou = 0.35

# batch_size = 128
# num_epochs = 25

test_images_path = '../../validare/validare'

# ------------------------------ run with train  ------------------------------

# print('dad')
# pos_desc_dad = get_positive_character_hog(character='dad', dim_window=dim_window, dim_hog_cell=dim_hog_cell, augument=True)
# neg_desc_dad = get_negative_character_hog(character='dad', dim_window=dim_window, dim_hog_cell=dim_hog_cell, neg_per_image=neg_per_image, min_patch_size=min_patch_size, max_patch_size=max_patch_size, max_iou=max_iou)
# np.save('dad_pos.npy', pos_desc_dad)
# np.save('dad_neg.npy', neg_desc_dad)
# pos_desc_dad = None
# neg_desc_dad = None
# model_dad = get_model('dad_neg.npy', 'dad_pos.npy', batch_size=batch_size, num_epochs=num_epochs)
# save_model(model_dad, 'model_dad.pth')
# detections_dad, scores_dad, file_names_dad = run(model=model_dad, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)

# print('mom')
# pos_desc_mom = get_positive_character_hog(character='mom', dim_window=dim_window, dim_hog_cell=dim_hog_cell, augument=True)
# neg_desc_mom = get_negative_character_hog(character='mom', dim_window=dim_window, dim_hog_cell=dim_hog_cell, neg_per_image=neg_per_image, min_patch_size=min_patch_size, max_patch_size=max_patch_size, max_iou=max_iou)
# np.save('mom_pos.npy', pos_desc_mom)
# np.save('mom_neg.npy', neg_desc_mom)
# pos_desc_mom = None
# neg_desc_mom = None
# model_mom = get_model('mom_neg.npy', 'mom_pos.npy', batch_size=batch_size, num_epochs=num_epochs)
# save_model(model_mom, 'model_mom.pth')
# detections_mom, scores_mom, file_names_mom = run(model=model_mom, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)#

# print('dexter')
# pos_desc_dexter = get_positive_character_hog(character='dexter', dim_window=dim_window, dim_hog_cell=dim_hog_cell, augument=True)
# neg_desc_dexter = get_negative_character_hog(character='dexter', dim_window=dim_window, dim_hog_cell=dim_hog_cell, neg_per_image=neg_per_image, min_patch_size=min_patch_size, max_patch_size=max_patch_size, max_iou=max_iou)
# np.save('dexter_pos.npy', pos_desc_dexter)
# np.save('dexter_neg.npy', neg_desc_dexter)
# pos_desc_dexter = None
# neg_desc_dexter = None
# model_dexter = get_model('dexter_neg.npy', 'dexter_pos.npy', batch_size=batch_size, num_epochs=num_epochs)
# save_model(model_dexter, 'model_dexter.pth')
# detections_dexter, scores_dexter, file_names_dexter = run(model=model_dexter, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)

# print('deedee')
# pos_desc_deedee = get_positive_character_hog(character='deedee', dim_window=dim_window, dim_hog_cell=dim_hog_cell, augument=True)
# neg_desc_deedee = get_negative_character_hog(character='deedee', dim_window=dim_window, dim_hog_cell=dim_hog_cell, neg_per_image=neg_per_image, min_patch_size=min_patch_size, max_patch_size=max_patch_size, max_iou=max_iou)
# np.save('deedee_pos.npy', pos_desc_deedee)
# np.save('deedee_neg.npy', neg_desc_deedee)
# pos_desc_deedee = None
# neg_desc_deedee = None
# model_deedee = get_model('deedee_neg.npy', 'deedee_pos.npy', batch_size=batch_size, num_epochs=num_epochs)
# save_model(model_deedee, 'model_deedee.pth')
# detections_deedee, scores_deedee, file_names_deedee = run(model=model_deedee, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)

# np.save('fisiere_solutie/task2/detections_dad', detections_dad)
# np.save('fisiere_solutie/task2/detections_deedee', detections_deedee)
# np.save('fisiere_solutie/task2/detections_dexter', detections_dexter)
# np.save('fisiere_solutie/task2/detections_mom', detections_mom)
# np.save('fisiere_solutie/task2/file_names_dad', file_names_dad)
# np.save('fisiere_solutie/task2/file_names_deedee', file_names_deedee)
# np.save('fisiere_solutie/task2/file_names_dexter', file_names_dexter)
# np.save('fisiere_solutie/task2/file_names_mom', file_names_mom)
# np.save('fisiere_solutie/task2/scores_dad', scores_dad)
# np.save('fisiere_solutie/task2/scores_deedee', scores_deedee)
# np.save('fisiere_solutie/task2/scores_dexter', scores_dexter)
# np.save('fisiere_solutie/task2/scores_mom', scores_mom)

# ------------------------------ run test  ------------------------------

os.makedirs('../fisiere_solutie/task2', exist_ok=True)

print('dad')
model = load_model('model_dad.pth', input_length=1764)
detections_dad, scores_dad, file_names_dad = run(test_images_path=test_images_path, model=model, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)
np.save('../fisiere_solutie/task2/detections_dad', detections_dad)
np.save('../fisiere_solutie/task2/file_names_dad', file_names_dad)
np.save('../fisiere_solutie/task2/scores_dad', scores_dad)


print('deedee')
model = load_model('model_deedee.pth', input_length=1764)
detections_deedee, scores_deedee, file_names_deedee = run(test_images_path=test_images_path, model=model, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)
np.save('../fisiere_solutie/task2/detections_deedee', detections_deedee)
np.save('../fisiere_solutie/task2/file_names_deedee', file_names_deedee)
np.save('../fisiere_solutie/task2/scores_deedee', scores_deedee)


print('dexter')
model = load_model('model_dexter.pth', input_length=1764)
detections_dexter, scores_dexter, file_names_dexter = run(test_images_path=test_images_path, model=model, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)
np.save('../fisiere_solutie/task2/detections_dexter', detections_dexter)
np.save('../fisiere_solutie/task2/file_names_dexter', file_names_dexter)
np.save('../fisiere_solutie/task2/scores_dexter', scores_dexter)


print('mom')
model = load_model('model_mom.pth', input_length=1764)
detections_mom, scores_mom, file_names_mom = run(test_images_path=test_images_path, model=model, score_threshold=score_threshold, scale_factor=scale_factor, dim_window=dim_window, dim_hog_cell=dim_hog_cell)
np.save('../fisiere_solutie/task2/detections_mom', detections_mom)
np.save('../fisiere_solutie/task2/file_names_mom', file_names_mom)
np.save('../fisiere_solutie/task2/scores_mom', scores_mom)
