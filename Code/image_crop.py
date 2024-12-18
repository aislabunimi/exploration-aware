import cv2
import numpy as np


def crop_image(img):
    white_pixels_before = np.sum(img == 254)
    total_pixels_before = img.size
    ratio_before = white_pixels_before / total_pixels_before
    # Initialize boundary variables
    # 'left_max', 'right_min', 'top_max', and 'bottom_min' will hold the cropping boundaries
    left_max = 0
    right_min = img.shape[1] - 1  # The rightmost column
    top_max = 0
    bottom_min = img.shape[0] - 1  # The bottommost row

    # Calculate the final horizontal and vertical dimensions after cropping
    hor_fin = right_min - left_max
    ver_fin = bottom_min - top_max

    # Cropping the image to its borders
    # -----------------------------------------------------------------
    
    # Loop to determine the left boundary for cropping
    while left_max < img.shape[1] and (254 not in img.T[left_max]) and (0 not in img.T[left_max]):
        left_max += 100  # Increment by 100 pixels
    left_max = max(0, left_max - 100)  # Adjust and ensure it doesn't go below 0

    # Loop to determine the right boundary for cropping
    while right_min >= 0 and (254 not in img.T[right_min]) and (0 not in img.T[right_min]):
        right_min -= 100  # Decrement by 100 pixels
    right_min = min(img.shape[1] - 1, right_min + 100)  # Ensure it's within bounds

    # Loop to determine the top boundary for cropping
    while top_max < img.shape[0] and (254 not in img[top_max]) and (0 not in img[top_max]):
        top_max += 100  # Increment by 100 pixels
    top_max = max(0, top_max - 100)  # Ensure it doesn't go below 0

    # Loop to determine the bottom boundary for cropping
    while bottom_min >= 0 and (254 not in img[bottom_min]) and (0 not in img[bottom_min]):
        bottom_min -= 100  # Decrement by 100 pixels
    bottom_min = min(img.shape[0] - 1, bottom_min + 100)  # Ensure it's within bounds

    # Crop the image based on the calculated boundaries
    hor_fin = right_min - left_max
    ver_fin = bottom_min - top_max
    crop_image = img[top_max:bottom_min + 1, left_max:right_min + 1]

    #Adjustment to keep to final image squared
    #-----------------------------------------
    
    if hor_fin >= ver_fin:
        grey_mat = np.full((100, hor_fin+1), 205)
        flag = 0
        while ver_fin < hor_fin:
            if flag == 0:
                crop_image = np.concatenate((grey_mat, crop_image))
                flag = 1
            else:
                crop_image = np.concatenate((crop_image, grey_mat))
                flag = 0
            ver_fin = ver_fin + 100

    if hor_fin <= ver_fin:
        grey_mat=np.full((ver_fin + 1, 100), 205)
        flag = 0
        while ver_fin > hor_fin:
            if flag == 0:
                crop_image = np.concatenate((grey_mat,crop_image),axis=1)
                flag=1
            else:
                crop_image = np.concatenate((crop_image,grey_mat),axis=1)
                flag=0
            hor_fin=hor_fin+100

    # Return the cropped image
    white_pixels_after = np.sum((crop_image == 254) | (crop_image == 255))
    total_pixels_after = crop_image.size
    ratio_after = white_pixels_after / total_pixels_after
    ratio = ratio_before / ratio_after
    return crop_image