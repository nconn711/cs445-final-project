import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import cv2
import numpy as np

from model import Net
from preprocess import segment_hand


# Load the model
model_path = 'models/model_v0.pt'
model = Net()
model.load_state_dict(torch.load(model_path))
model.train(False)

def classify(frame):

    transform = transforms.Compose([
        transforms.ToTensor()  # convert to tensor
    ])

    with torch.no_grad():
        output = model(transform(frame))
        _, predicted = torch.max(output.data, -1)

    return predicted.item()


cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Resize the frame to 128 by 128 pixels
    frame = cv2.resize(frame, (128, 128))

    # Process the frame using the hand segmentation algorithm
    result = segment_hand(frame)
    prediction = classify(result)

    # Add a third dimension for the channel if it is missing
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Resize the result back to the original size
    result = cv2.resize(result, (frame.shape[1], frame.shape[0]))

    # Concatenate the original frame and the processed result horizontally
    output = np.concatenate((frame, result), axis=1)
    cv2.putText(output, str(prediction), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    # Display the output
    cv2.imshow('Hand Segmentation', output)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()