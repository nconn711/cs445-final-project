import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import cv2
import numpy as np
from skimage import measure


def segment_hand(frame):
    frame = cv2.resize(frame, (128, 128))
    blur = cv2.medianBlur(frame, 3)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # apply Otsu's thresholding to the 'a' channel
    _, thresh = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # perform a morphological opening operation to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    labels = measure.label(morph, connectivity=2)

    # find the largest connected component and create a mask for it
    mask = np.zeros(morph.shape, dtype="uint8")
    largest_component = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = np.zeros(morph.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        if num_pixels > largest_component:
            largest_component = num_pixels
            mask = label_mask

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.resize(result, (frame.shape[1], frame.shape[0]))

    return mask
