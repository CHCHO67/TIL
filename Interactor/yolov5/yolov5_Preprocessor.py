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
from Adaptor.yolov5_Capture import LoadImages, LoadStreams 


class Preprocessor():
    def __init__(self, opt):
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(('rtsp://', 'rtmpß://', 'http://'))
        self.device = select_device(opt.device)

    def __loadDataset__(self):
        # Initialize
        half = self.device.type != 'cpu'

        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(self.imgsz, s=model.stride.max())  # check img_size
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
        
    def __getParameter__(opt):
        # Directories
        save_img = True
        view_img = False
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        device = select_device(opt.device)
        half = device.type!= 'cpu'
        vid_path, vid_writer = None, None

        # Second-stage classifier
        model = attempt_load(opt.weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
                         
        if half:
            model.half()  # to FP16

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]    

        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
            
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        


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