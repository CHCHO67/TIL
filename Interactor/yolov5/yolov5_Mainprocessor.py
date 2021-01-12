import argparse
import time
from pathlib import Path

import sys
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized

from yolov5_Preprocessor import Preprocessor
from yolov5_Postprocessor import Postprocessor


class Mainprocessor():
    def __init__(self):
        self.webcam = False
        # Initialize
        set_logging()
        # print model summmary - ex)Model Summary: 484 layers, 88397343 parameters, 0 gradients
        
        print("main_init")
        

    def __detect__(self, dataset, opt, save_img):
                
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16
            
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            
        if save_img:
            view_img = False
            
        postprocessor = Postprocessor(model)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                    img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            print("2")
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                print("3")
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                postprocessor.__rescaleBoxes__(det, s, img, im0, opt.save_txt,opt.save_conf,save_img, view_img)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                postprocessor.__getResult__(dataset,vid_cap, p, view_img, save_img, save_path, im0)
                    
        postprocessor.__resultInfo__(save_txt, save_img, save_dir)
        
        print(f'Done. ({time.time() - t0:.3f}s)')
'''
if __name__ == '__main__':
    cap = Capture()
    source, imgsz = cap.__capture__()

    pre = Preprocessor(source, imgsz)
    dataset = pre.__preprocessor__()
    
    main = Mainprocessor(dataset,imgsz)
    det = main.__mainprocessor__()
    print("end of code")
'''


        

            