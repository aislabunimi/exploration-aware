import os
import cv2
from natsort import natsorted
import numpy as np
import csv
from multiprocessing import Pool, Manager, Value
import ctypes
import math
from datetime import datetime

ls = []

dataset_dir = "/PATH_TO_FOLDER_WITH_MAP_IMAGES/output"
snapshots_save_rate = 600
# envs_path=os.path.join(os.getcwd(),'GradCAM_predictions_sorted','grad_sorted_1')
envs_path = "Results/GradCAM_predictions/labels_for_metrics_KTH+MIT"
cwd = os.getcwd()


# envs_path='/media/mrk/TOSHIBA EXT/pc/TESI_REPO/tesi.ferrara.earlystopping/TESI/ROS/src/slampbenchmarking/runs/test_grad_enet_500_full/'
def extract_env(img_name):
    env = img_name.split('@')[0]
    return env


def extract_run(img_name):
    run = img_name.split('@')[1]
    return run


def extract_timestamp(img_name):
    timestamp = img_name.split('@')[2]
    timestamp = int(float(timestamp[:-7]))
    return timestamp


def extract_minutes(seconds):
    minutes = float(seconds) / 60
    minutes_10 = 10 * math.ceil(float(minutes) / 10)
    return int(minutes_10)


def pixel_difference(image, ground_truth):
    im1 = cv2.imread(image)
    im2 = cv2.imread(ground_truth)

    GT_pixels = np.where(im2 == 254)
    GT_pixels = GT_pixels[1].size
    diff_pixels = np.where(im1 != im2)
    diff_pixels = diff_pixels[1].size
    difference_percentage = (diff_pixels / GT_pixels) * 100
    return difference_percentage


def get_saved_time():
    filename = datetime.now().strftime('saved_time_offline_%d%m%Y_%H%M.csv')
    csv_path = os.path.join(cwd, 'Results', 'Saved_time_offline', filename)
    print(csv_path)
    f = open(csv_path, 'w')
    writer = csv.writer(f)
    header = ['img_id', 'GT_timestamp', 'saved_time%_TP', 'timestamp_FPvsTP', 'saved_time%_withFP', 'confusion_matrix']
    writer.writerow(header)
    runs_number = 0
    tot_runs = 0
    tot_seconds_baseline = 0
    tot_seconds_saved_TP = 0
    tot_seconds_saved_FP = 0
    tot_seconds_lost_FN = 0
    tot_perc_seconds_saved = 0
    tot_perc_seconds_saved_with_FP = 0
    tot_perc_seconds_saved_with_FN = 0
    fp_number = 0
    tp_number = 0
    tn_number = 0
    fn_number = 0
    fp_and_fn = 0

    tmp_env = ''
    print(natsorted(os.listdir(envs_path)))
    for env in natsorted(os.listdir(envs_path)):
        tmp_env = env
        runs_path = os.path.join(envs_path, env)
        for run in natsorted(os.listdir(runs_path)):
            TP_path = os.path.join(runs_path, run, 'EXPLORED_E')
            TN_path = os.path.join(runs_path, run, 'NOT_EXPLORED_NE')
            FP_path = os.path.join(runs_path, run, 'EXPLORED_NE')
            FN_path = os.path.join(runs_path, run, 'NOT_EXPLORED_E')
            TP_len = len(natsorted(os.listdir(TP_path)))
            TN_len = len(natsorted(os.listdir(TN_path)))
            FP_len = len(natsorted(os.listdir(FP_path)))
            FN_len = len(natsorted(os.listdir(FN_path)))
            runs_number += 1
            tot_runs = tot_runs + 1

            if (len(natsorted(os.listdir(TP_path))) == 0):
                if (len(natsorted(os.listdir(FN_path))) == 0):
                    tot_runs -= 1
                else:
                    tmp_id = env + '@' + run + '@NO_TPs'

                    row = [tmp_id, str(TP_len) + ' True Positives', str(TN_len) + ' True Negatives',
                           str(FP_len) + ' False Positives', str(FN_len) + ' False Negatives']
                    writer.writerow(row)

            for image in natsorted(os.listdir(TP_path)):
                # print('IG= ',image)
                timestamp = extract_timestamp(image)
                print(f"timestamp: {timestamp}")
                minutes = extract_minutes(timestamp)
                print(f"minutes: {minutes}")
                seconds_saved = (len(os.listdir(TP_path)) - 1) * 10  # *snapshots_save_rate
                timestamp_gt = extract_timestamp((natsorted(os.listdir(TP_path)))[-1])
                if timestamp_gt == 0:
                    print(f"timestamp_gt for {TP_path} is zero ......")
                    continue
                print(f"timestamp_gt: {timestamp_gt}")
                perc_seconds_saved = ((timestamp_gt - timestamp) / timestamp_gt) * 100
                orig_img_dir = os.path.join(dataset_dir, env, run, 'Maps')
                orig_img_name = os.path.join(orig_img_dir, str(minutes) + 'Map.png')
                # print("Timestamp=", timestamp)
                # print("Timestamp_gt=", timestamp_gt)
                # print("saved=",seconds_saved)

                # print("savedprc=",perc_seconds_saved)
                # print('\n')
                tot_seconds_baseline += timestamp_gt
                tot_seconds_saved_TP += seconds_saved
                img_id = env + '@' + run + '@' + image
                # row= [img_id,timestamp_gt,perc_seconds_saved]
                # writer.writerow(row)

                if len(os.listdir(FP_path)) != 0:
                    timestamp_FP = extract_timestamp((natsorted(os.listdir(FP_path)))[0])
                    perc_seconds_saved_with_FP = ((timestamp_gt - timestamp_FP) / timestamp_gt) * 100
                    tot_perc_seconds_saved_with_FP += perc_seconds_saved_with_FP
                    row = [img_id, timestamp_gt, perc_seconds_saved, str(timestamp_FP) + 'vs' + str(timestamp),
                           perc_seconds_saved_with_FP,
                           str(TP_len) + ' TP/ ' + str(TN_len) + ' TN/ ', str(FP_len) + ' FP/ ' + str(FN_len) + ' FN']
                    writer.writerow(row)
                else:
                    row = [img_id, timestamp_gt, perc_seconds_saved, 0, perc_seconds_saved,
                           str(TP_len) + ' TP/ ' + str(TN_len) + ' TN/ ', str(FP_len) + ' FP/ ' + str(FN_len) + ' FN']
                    writer.writerow(row)
                    tot_perc_seconds_saved_with_FP += perc_seconds_saved
                if len(os.listdir(FN_path)) != 0:
                    timestamp_FN = extract_timestamp((natsorted(os.listdir(FN_path)))[0])
                    tot_perc_seconds_saved_with_FN += ((timestamp_gt - timestamp_FN) / timestamp_gt) * 100
                else:
                    tot_perc_seconds_saved_with_FN += perc_seconds_saved
                tot_perc_seconds_saved += perc_seconds_saved
                tp_number += len(os.listdir(TP_path))
                break

            for image in natsorted(os.listdir(FP_path)):
                timestamp = extract_timestamp(image)
                minutes = extract_minutes(timestamp)
                seconds_saved = (len(os.listdir(FP_path))) * snapshots_save_rate
                timestamp_gt = extract_timestamp((natsorted(os.listdir(FP_path)))[-1])
                orig_img_dir = os.path.join(dataset_dir, env, run, 'Maps')
                orig_img_name = os.path.join(orig_img_dir, str(minutes) + 'Map.png')
                fp_number += len(os.listdir(FP_path))
                tot_seconds_saved_FP += seconds_saved
                break

            for image in natsorted(os.listdir(TN_path)):
                timestamp = extract_timestamp(image)
                minutes = extract_minutes(timestamp)
                tn_number += len(os.listdir(TN_path))
                break

            for image in natsorted(os.listdir(FN_path)):
                timestamp = extract_timestamp(image)
                minutes = extract_minutes(timestamp)
                seconds_lost = (len(os.listdir(FN_path))) * snapshots_save_rate
                fn_number += len(os.listdir(FN_path))
                tot_seconds_lost_FN += seconds_lost
                break

            if len(os.listdir(FP_path)) != 0 and len(os.listdir(FN_path)) != 0:
                fp_and_fn += 1
                ls.append([env, run, len(os.listdir(FP_path)), len(os.listdir(FN_path))])

        writer.writerow(['number_of_runs', runs_number])
        writer.writerow(['AVG_PERC_SEC_SAVED_WITH_TP', tot_perc_seconds_saved / runs_number])
        writer.writerow(['AVG_PERC_SEC_SAVED_WITH_FP', tot_perc_seconds_saved_with_FP / runs_number])
        writer.writerow(['AVG_PERC_SEC_SAVED_WITH_FN', tot_perc_seconds_saved_with_FN / runs_number])

    print('RUNS_NUMBER ', runs_number, tot_runs)
    print('TOT_SECS_BASELINE ', tot_seconds_baseline)
    print('TOT_SECS_SAVED_TP ', tot_seconds_saved_TP)
    # print('TOT_SECS_SAVED_FP ',tot_seconds_saved_FP)
    # print('TOT_SECS_LOST_FN ',tot_seconds_lost_FN)
    print('AVG_PERC_SEC_SAVED', tot_perc_seconds_saved / runs_number)
    print('AVG_PERC_SEC_SAVED_WITH_FP', tot_perc_seconds_saved_with_FP / runs_number)
    print('AVG_PERC_SEC_SAVED_WITH_FN', tot_perc_seconds_saved_with_FN / runs_number)

    # print('AVG_SECS_SAVED= ',tot_seconds_saved_TP/runs_number)
    # print('AVG_SECS_SAVED_WITH_FP= ',(tot_seconds_saved_FP+tot_seconds_saved_TP)/runs_number)
    # print('AVG_MINS_SAVED= ',(tot_seconds_saved_TP/runs_number)/60)
    # print('AVG_MINS_SAVED_WITH_FP= ',((tot_seconds_saved_FP+tot_seconds_saved_TP)/runs_number)/60)
    # print('AVG_SECS_LOST= ',tot_seconds_lost_FN/runs_number)

    print('TP_NUMBER= ', tp_number)
    print('FP_NUMBER= ', fp_number)
    print('TN_NUMBER= ', tn_number)
    print('FN_NUMBER= ', fn_number)
    # 30% every 10 minutes/40%every 30 secs
    tmp_val = tot_perc_seconds_saved / runs_number

    # print( tmp_env + ' & ' + str(runs_number)  + ' & ' + str(tmp_val)[0:5]  + '\% & ' + str(tot_seconds_baseline) + ' & ' + str(tot_seconds_saved_TP))
    print("BOTH FP AND FN: ", fp_and_fn, " times")
    print(ls)


if __name__ == '__main__':
    get_saved_time()
