from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from sort import iou, Tracklet
from detector import Detector

import numpy as np
import os
import sys
import time
import datetime
import argparse

import cv2
import torch
import imutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csrt",         type=bool,  default=False,                      help="use csrt tracker")
    parser.add_argument("--image_folder", type=str,   default="data/samples")
    parser.add_argument("--model_def",    type=str,   default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str,   default="weights/yolov3_v2.pth",    help="path to weights file")
    parser.add_argument("--class_path",   type=str,   default="config/classes.names",     help="path to class label file")
    parser.add_argument("--conf_thres",   type=float, default=0.6,                        help="object confidence threshold")
    parser.add_argument("--nms_thres",    type=float, default=0.1, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size",     type=int,   default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    if opt.csrt:
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
        }
        trackers = cv2.MultiTracker_create()

    yolo = Detector(opt.model_def, opt.weights_path, opt.class_path, opt.conf_thres, opt.nms_thres, opt.img_size)
    tracklets = []
    object_id = 0

    for i in range(0, 76):
        frame = cv2.imread(os.path.join(opt.image_folder, "frame{:04d}.jpg".format(i)))
        if opt.csrt:
            frame = imutils.resize(frame, width=400)

        yolo_boxes = yolo.detect(frame)

        for tr in tracklets:
            tr.setActive(False)

        for yolo_box in yolo_boxes:
            match = 0
            for tr in tracklets:
                if yolo_box[-1] != tr.label:
                    continue

                if iou(yolo_box[:4], tr.getState()):
                    tr.update(yolo_box[:4])
                    match = 1
                    tr.setActive(True)
                    break

            if match == 0:
                tr = Tracklet(object_id, yolo_box[-1], yolo_box[:-1])
                tracklets.append(tr)
                if opt.csrt:
                    tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
                    trackers.add(tracker, frame, (yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3]))

                object_id += 1

        for tr in tracklets:
            if tr.active==False:
                tr.addTimeout()
            if tr.timeout > 10:
                tracklets.remove(tr)
                continue

            if tr.active:
                (x, y, w, h) = tr.getState()
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), tr.color, 2)

            if opt.csrt:
                (success, csrt_boxes) = trackers.update(frame)
                tr.predict(csrt_boxes[tr.idx])
            else:
                tr.predict()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()

