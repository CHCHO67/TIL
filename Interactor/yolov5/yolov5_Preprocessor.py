import argparse
import time
from pathlib import Path

import os
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Adaptor.yolov5_Capture import Capture 

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class Preprocessor():
    def __init__(self, opt):
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(('rtsp://', 'rtmpß://', 'http://'))
        self.device = select_device(opt.device)

    def __preprocessor__(self):
        # Initialize
        half = self.device.type != 'cpu'

         # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        if self.webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz)
            return dataset, view_img
        else:
            save_img = True
            dataset = LoadImages(self.source, img_size=self.imgsz)
            return dataset, save_img

        #test print
        '''
        for path,img,im0s,vid_cap in dataset:
            print(path)
            print(img)
            print("\n")
            print(im0s)
            print("\n")
            print(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        '''