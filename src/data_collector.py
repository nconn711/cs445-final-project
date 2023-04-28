import cv2
import numpy as np

from model import Net
from preprocess import segment_hand


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

    # Wait for a key press and get the key code
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Exit if the 'q' key is pressed
        break
    elif ord('0') <= key <= ord('5'):
        # Set the number based on the key pressed
        number = key - ord('0')


# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
