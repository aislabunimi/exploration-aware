import cv2
import os
import shutil
import pandas as pd
from os.path import join
import numpy as np
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CSV_DATA_DIR, FULL_SIZE_RAW_MAPS_DIR
from loguru import logger
import natsort

def process_snapshots(snapshots: list[str], basePath: str, area_dict: dict) -> None:
    last_img = cv2.imread(join(basePath, snapshots[-1]), cv2.IMREAD_GRAYSCALE)
    GT_pixels = np.sum(last_img == 254)
    for snap in snapshots:
        img = cv2.imread(join(basePath, snap), cv2.IMREAD_GRAYSCALE)
        diff_pixels = np.sum(img != last_img)
        area_dict[snap] = float(diff_pixels / GT_pixels) * 100

def process_runs(runs: list[str], basePath: str, area_dict: dict) -> None:
    get_run_id = lambda x : x.split('@')[1]
    all_runs = set((get_run_id(run) for run in runs))
    for run in all_runs:
        snapshots = [img for img in runs if get_run_id(img) == run] # get all snapshots of a single run
        snapshots = natsort.natsorted(snapshots)
        process_snapshots(snapshots, basePath, area_dict)

def create_regression_dataFrame(dirPath: str) -> pd.DataFrame:
    # name format is 'Map_name @ run_id @ snapshot'
    get_map_name = lambda fName: fName.split('@')[0] # extract only the map Name before the '@'
    all_maps = set((get_map_name(fName) for fName in os.listdir(dirPath))) # get unique maps in the folder
    area_dict: dict[str, float] = {}
    for map in all_maps:
        runs = [img for img in os.listdir(dirPath) if img.startswith(map)]  # get all the runs of a single map
        process_runs(runs, dirPath, area_dict)
    
    return pd.DataFrame(area_dict.items(), columns=['id', 'area_perc']).sort_values(by='id').reset_index(drop=True)

def create_classification_dataFrame(dirPath: str) -> pd.DataFrame:
    data_split = dirPath.split('/')[-1] # identify train|test|valid split
    dest_folder = join(PROCESSED_DATA_DIR, data_split)
    os.makedirs(dest_folder, exist_ok=True)
    img_dict: dict[str, int] = {}       # create a dict with binary labels for classification
    for root_folder, dirs, files in os.walk(dirPath):
        if dirs:
            continue
        for img in files:
            img_dict[img] = 0 if root_folder.endswith('NOT_EXPLORED') else 1

        shutil.copytree(root_folder, dest_folder, dirs_exist_ok=True)

        logger.info(f"Copying {data_split}/{root_folder.split('/')[-1]} into {dest_folder}")

    return pd.DataFrame(img_dict.items(), columns=['id', 'explored']).sort_values(by='id').reset_index(drop=True)

if __name__ == "__main__":
    splits = ("test", "train", "valid")
    # Check we're in the same directory as the splits
    assert set(splits).issubset(os.listdir(RAW_DATA_DIR))
    
    for split in splits:
        df_class = create_classification_dataFrame(str(RAW_DATA_DIR / split))
        df_regr = create_regression_dataFrame(str(FULL_SIZE_RAW_MAPS_DIR / split))
        
        df_class['area_perc'] = df_regr['area_perc']    # Add the two columns
        os.makedirs(CSV_DATA_DIR, exist_ok=True)
        df_class.to_csv(CSV_DATA_DIR / f'df_{split}.csv', index=False)
        logger.info(f"Saving df_{split}.csv to {CSV_DATA_DIR}")

