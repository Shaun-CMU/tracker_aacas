# tracker_aacas
tracker testing for aacas

## node.py
Main process of SORT tracking based on CSRT and YOLO detections.

`$ python3 node.py`

## detector.py
Initializes YOLO model and detection process of each frame.

## sort.py
Handles object tracklets and Kalman filters.

## models.py
Defines YOLO model architecture.
