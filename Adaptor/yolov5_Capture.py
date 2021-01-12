import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import sys

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Capture:
    def __init__(self,opt):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    def __capture__(self):
        # Initialize
        #set_logging()
        # print model summmary - ex)Model Summary: 484 layers, 88397343 parameters, 0 gradients

        #device = select_device(opt.device)
        #half = device.type != 'cpu'  # half precision only supported on CUDA

         # Load model
        model = attempt_load(self.weigths, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(self.imgsz, s=model.stride.max())  # check img_size

        # Set Dataloader
        vid_path, vid_writer = None, None
        '''
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
        '''
        save_img = True
        #dataset = LoadImages(self.source, img_size=imgsz)
        print(self.source, self.imgsz)
   
        #print(type(dataset))

        return self.source, self.imgsz

#get Input RGB Frame in dataset
'''
if __name__ == '__main__':
    opt=None
    cap = Capture()
    dataset = cap.__capture__()
    '''