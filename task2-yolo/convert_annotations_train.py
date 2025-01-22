import os
import shutil

input_base_dir = "../../antrenare"
output_images_dir = "images/train"
output_labels_dir = "labels/train"

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)


def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height


image_counter = 1

for character in ["dad", "deedee", "dexter", "mom"]:
    annotation_file = os.path.join(input_base_dir, f"{character}_annotations.txt")
    image_folder = os.path.join(input_base_dir, character)

    with open(annotation_file, "r") as af:
        annotations = af.readlines()

    image_annotations = {}
    for annotation in annotations:
        parts = annotation.strip().split()
        image_name, xmin, ymin, xmax, ymax, label = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), parts[5]

        if image_name not in image_annotations:
            image_annotations[image_name] = []
        image_annotations[image_name].append((xmin, ymin, xmax, ymax, label))

    for image_name, boxes in image_annotations.items():

        new_image_name = f"{image_counter:04d}.jpg"
        output_image_path = os.path.join(output_images_dir, new_image_name)
        output_label_path = os.path.join(output_labels_dir, new_image_name.replace(".jpg", ".txt"))

        input_image_path = os.path.join(image_folder, image_name)
        shutil.copy(input_image_path, output_image_path)

        img_width, img_height = 480, 360

        with open(output_label_path, "w") as olf:
            for xmin, ymin, xmax, ymax, label in boxes:
                x_center, y_center, width, height = convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height)
                if label in ["dad", "deedee", "dexter", "mom"]:
                    num_label = ["dad", "deedee", "dexter", "mom"].index(label)
                    olf.write(f"{num_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        image_counter += 1

print("Processing complete. Images and labels saved.")

