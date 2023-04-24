import sys

import cv2
import numpy as np

from train_model import Net
from data_preprocess import segment_hand


number = 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (128, 128))

    # Process the frame using the hand segmentation algorithm
    result = segment_hand(frame)

    # TODO: store result as a image with label *number*

    # Add a third dimension for the channel if it is missing
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    result = cv2.resize(result, (frame.shape[1], frame.shape[0]))

    # Concatenate the original frame and the processed result horizontally
    output = np.concatenate((frame, result), axis=1)
    cv2.putText(output, str(number), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.imshow('Hand Segmentation', output)

    # Exit if the 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('0'):
        number = 0
    elif key == ord('1'):
        number = 1
    elif key == ord('2'):
        number = 2
    elif key == ord('3'):
        number = 3
    elif key == ord('4'):
        number = 4
    elif key == ord('5'):
        number = 5

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()