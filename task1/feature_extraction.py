import numpy as np
import glob
import cv2 as cv
import os
import ntpath
from skimage.feature import hog
from tqdm import trange
from helpers import *


def get_positive_hog(dim_window=36, dim_hog_cell=6, augument=False):

    characters = ['dad', 'deedee', 'dexter', 'mom']
    positive_descriptors = []

    for i in trange(len(characters)):

        character = characters[i]
        images_path = os.path.join('../../antrenare/' + character + '/', '*.jpg')
        bounding_boxes = np.loadtxt('../../antrenare/' + character + '_annotations.txt', dtype='str')

        images_path_list = glob.glob(images_path)

        for path in images_path_list:
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            short_file_name = ntpath.basename(path)
            annotations = bounding_boxes[bounding_boxes[:, 0] == short_file_name]

            for annotation in annotations:
                face = image[int(annotation[2]):int(annotation[4]), int(annotation[1]):int(annotation[3])].copy()
                face = cv.resize(face, (dim_window, dim_window))
                features = hog(face, pixels_per_cell=(dim_hog_cell, dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)

                if augument:
                    features = hog(np.fliplr(face), pixels_per_cell=(dim_hog_cell, dim_hog_cell),
                                   cells_per_block=(2, 2), feature_vector=True)
                    positive_descriptors.append(features)

    positive_descriptors = np.array(positive_descriptors)

    return positive_descriptors


def get_negative_hog(dim_window=36, dim_hog_cell=6, neg_per_image=6, min_patch_size=32, max_patch_size=120, max_iou=0.25):

    characters = ['dad', 'deedee', 'dexter', 'mom']
    negative_descriptors = []

    for i in trange(len(characters)):

        character = characters[i]
        images_path = '../../antrenare/' + character + '/*.jpg'

        images_path_list = glob.glob(images_path)
        bounding_boxes = np.loadtxt('../../antrenare/' + character + '_annotations.txt', dtype='str')

        for path in images_path_list:

            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            num_rows, num_cols = image.shape
            cnt = neg_per_image

            short_file_name = ntpath.basename(path)
            annotations = bounding_boxes[bounding_boxes[:, 0] == short_file_name]

            while cnt:

                patch_size = np.random.randint(low=min_patch_size, high=max_patch_size)

                # topleft corner
                x = np.random.randint(low=0, high=num_cols - patch_size)
                y = np.random.randint(low=0, high=num_rows - patch_size)

                # bottomright corner
                x2 = x + patch_size
                y2 = y + patch_size

                valid_negative = 1

                for annotation in annotations:
                    if intersection_over_union([x, y, x2, y2],
                                               [int(annotation[1]), int(annotation[2]), int(annotation[3]), int(annotation[4])]) > max_iou:
                        valid_negative = 0
                        break

                if valid_negative:
                    patch = image[y: y2, x: x2]
                    cnt -= 1
                    resized_patch = cv.resize(patch, (dim_window, dim_window), interpolation=cv.INTER_AREA)
                    descr = hog(resized_patch, pixels_per_cell=(dim_hog_cell, dim_hog_cell),
                                cells_per_block=(2, 2), feature_vector=False)
                    negative_descriptors.append(descr.flatten())

    negative_descriptors = np.array(negative_descriptors)

    return negative_descriptors
