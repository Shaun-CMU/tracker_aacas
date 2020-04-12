#!/usr/bin/env python

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from sort import iou, Tracklet
from detector import Detector

import cv2
import numpy as np
import os, sys

import torch
import rospy
from rospkg import RosPack

package = RosPack()
package_path = package.get_path('yolov3_sort')

class Tracker():
    def __init__(self):
        weights_name = rospy.get_param('~weights_name', 'yolov3_v2.pth')
        self.weights_path = os.path.join(package_path, 'weights', weights_name)
        if not os.path.isfile(self.weights_path):
            raise IOError("weights not found :(")

        config_name = rospy.get_param('~config_name', 'yolov3-custom.cfg')
        self.config_path = os.path.join(package_path, 'config', config_name)
        classes_name = rospy.get_param('~classes_name', 'custom.names')
        self.classes_path = os.path.join(package_path, 'config', classes_name)

        self.conf_thres = rospy.get_param('~confidence', 0.7)

        self.tracked_objects_topic = rospy.get_param('~tracked_objects_topic')

if __name__=="__main__":
    rospy.init_node("tracker_node")
    TR = Tracker()
