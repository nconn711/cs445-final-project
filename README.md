Hand Gesture Digit Classifier
=============================

This project is a neural network model that classifies hand gesture digits 0 through 5.

Data
----

The `data` directory contains the original dataset of hand gesture digits. The images are stored in subdirectories corresponding to their class label.

The `data_new` directory contains a new dataset of hand gesture digits. The images are stored in subdirectories corresponding to their class label.

Models
------

The `models` directory contains saved versions of trained models. The model filenames include a version number (`v0`, `v1`, `v2`) to track different iterations of the model.

Source code
-----------

The `src` directory contains the source code for the project.

-   `data_collector.py`: script for collecting new images.
-   `model.py`: definition of the neural network model architecture.
-   `preprocess.py`: definition of preprocessing functions for during data collection.
-   `run_model.py`: script for running the trained model on live video feed.
-   `train_model.py`: script for training the model.
