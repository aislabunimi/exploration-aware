import os
import glob
import random
import sys
import csv
import json
import multiprocessing
import time
from random import seed
from shutil import copy


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from natsort import natsorted


from support_funs import rotate_bound_modified
from cluster_pixels import cluster_pixels
from image_crop import crop_image

# Full unlabeled dataset directory

# new dataset
# dataset_dir = '/media/aislab/EXTERNAL_USB/Dati_Tesi_Princisgh/Online/KTH+MIT/output'

# old dataset
dataset_dir = '/media/michele/EXTERNAL_USB/Dati_Tesi_Princisgh/Online/KTH+MIT/partial'
office_env = 'office_c'

dataset_name = 'KTH+MIT'

# Directories where the dataset will be split
cwd = os.getcwd()
train_dir = os.path.join(cwd, 'Datasets/Labelled_Data/KTH+MIT/train/')
valid_dir = os.path.join(cwd, 'Datasets/Labelled_Data/KTH+MIT/valid/')
test_dir = os.path.join(cwd, 'Datasets/Labelled_Data/KTH+MIT/test/')

seed(1)


def folder_loop(env):
    if env != office_env:  # the office_c env is not saved correctly
        env_path = os.path.join(dataset_dir, env)
        runs_list = natsorted(os.listdir(env_path))
        pixel_ratios = []
        large_thresh = 0.16
        medium_thresh = 0.4

        if runs_list[-1] == 'z_resTests':
            runs_list = runs_list[:len(runs_list) - 1]

        for run in runs_list:

            # pngs = glob.glob(dataset_dir + '/' + env + '/'+run+"/**/Maps/*.png",recursive = True)
            pngs = glob.glob(os.path.join(dataset_dir, env, run, '**', 'Maps', '*.png'), recursive=True)
            pngs = natsorted(pngs)
            if pngs:
                gt_dir = pngs[-1]
                ground_truth = cv2.imread(gt_dir)

            else:
                print("No files in directory, cannot find ground truth")
                ground_truth = None

            # Loop over the images in a single run of a single environment
            # ------------------------------------------------------------
            for png in pngs:
                ratio = 0

                img = cv2.imread(png)

                if img is None:
                    print(f"Error loading image: {png}. Skipping...")

                # Extract the name of the image (just the name not the full path)
                # ---------------------------------------------------------------------------------------
                for k in range(len(png) - 1, 0, -1):
                    if png[k] == '/':
                        name = png[k + 1:]
                        break
                if name == "Map.png":
                    print("skipping Map.png")
                    continue
                name = env + '@' + run + '@' + name

                if ratio > large_thresh:
                    res = cluster_pixels(img, ground_truth, env, run, name, "large")

                elif large_thresh >= ratio >= medium_thresh:
                    res = cluster_pixels(img, ground_truth, env, run, name, "medium")
                else:
                    res = cluster_pixels(img, ground_truth, env, run, name, "large")

                try:

                    img = crop_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

                    # Convert image to uint8 for compatibility with OpenCV functions
                    img = img.astype(np.uint8)

                    # Resize the image
                    img = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)

                except cv2.error as e:
                    # Handle any OpenCV errors (e.g., cropping or resizing issues)
                    print(f"OpenCV error for image {png}: {e}")
                    continue

                # if env in env_dict:
                #     res = cluster_pixels(png, gt_dir, env, run, env_dict[env])
                # else:
                #     res = cluster_pixels(png, gt_dir, env, run, "medium")

                # ----------------------------------------------------------------------------------------

                # Create train/val/test directories
                # ----------------------------------------------

                # Determine directory and CSV file based on env
                if env in valid_env_list:
                    directory = valid_dir
                    csv_file = 'df_val.csv'
                elif env in test_env_list:
                    directory = test_dir
                    csv_file = 'df_test.csv'
                else:
                    directory = train_dir
                    csv_file = 'df_train.csv'

                # Determine subfolder and row data based on res[0]
                if res[0] in [0, 1]:
                    subfolder = 'EXPLORED' if res[0] == 1 else 'NOT_EXPLORED'
                    img_path = os.path.join(directory, subfolder, name)
                    row = [img_path, res[0], res[1]]

                    # Save image and write row to CSV file
                    cv2.imwrite(img_path, img)
                    with open(f'dataframes/{csv_file}', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)

                elif res[0] == 2:
                    print("skipping useless for regression")
            if env == env_folder_list[-1] and run == runs_list[-1] and png == pngs[-1]:
                plt.figure(figsize=(10, 6))
                plt.hist(res[2], bins=100, color='blue', alpha=0.7, edgecolor='black')
                plt.title('Distribution of pixels in clusters')
                plt.xlabel('Pixels per cluster')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.show()


# Prepares the 2 class folders(EXPLORED and NOT_EXPLORED) in the train/validation/test directories defined at the top.
# If those folders already exist the program ends (to avoid overwriting an existent set of images)
def folder_loop_prep():
    def create_dirs(base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(os.path.join(base_dir, 'EXPLORED'))
            os.makedirs(os.path.join(base_dir, 'NOT_EXPLORED'))
        else:
            print('Existent_folders')
            return False
        return True

    env_folder_list = natsorted(os.listdir(dataset_dir))

    # Check and create directories
    if not create_dirs(train_dir) or not create_dirs(test_dir) or not create_dirs(valid_dir):
        return

    return env_folder_list


def write_headers_once(file_path, headers):
    # Check if the file already exists and has content
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the headers only if the file doesn't exist or is empty
        if not file_exists:
            writer.writerow(headers)


def add_headers(filepath):
    df = pd.read_csv(filepath, header=None)
    df_headers = ['img_path', 'class', 'diff_perc']
    df.columns = df_headers
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    with open('subdirectories_mit_kth.txt') as f:
        env_dict = json.load(f)
    with open('test_env_list.txt', 'r') as f:
        test_env_list = f.read()  # + 'E18-5'

    csv_files = ['dataframes/df_train.csv', 'dataframes/df_test.csv', 'dataframes/df_val.csv']
    env_folder_list = folder_loop_prep()
    env_folder_list = natsorted(os.listdir(dataset_dir))

    # Define split ratios for the dataset
    train_ratio = 0.7  # 70% of data for training
    valid_ratio = 0.2  # 20% for validation

    # Randomly shuffle the environment list to ensure randomness in the split
    random.shuffle(env_folder_list)

    # Calculate the size of each split based on the defined ratios
    train_size = int(len(env_folder_list) * train_ratio)
    valid_size = int(len(env_folder_list) * valid_ratio)

    # Assign environments to training, validation, and testing sets
    train_env_list = env_folder_list[:train_size]
    valid_env_list = env_folder_list[train_size:train_size + valid_size]
    test_env_list = env_folder_list[train_size + valid_size:]
    pool = multiprocessing.Pool()
    pool.map(folder_loop, env_folder_list)
    list(map(add_headers, csv_files))



