import cv2
from tqdm import tqdm
import sys
import os

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from yolov5_Preprocessor import Preprocessor
from yolov5_Mainprocessor import Mainprocessor
#from Adaptor.VideoCapturer import videoCapturer
    

class AIanalyzer():
    def __init__(self, opt):
        #super().__init__(mode='video',path=video_path, save_dir=save_dir)
        #self.cap = Capture(opt)
        self.pre = Preprocessor(opt)
        dataset, sv_img = self.pre.__preprocessor__()
        self.main = Mainprocessor(dataset, opt)
        det = self.main.__mainprocessor__()
        #self.pro = Postproceesor()
        
        print("end of code")

    def __aianalyzer__(self):
        #source, imgsz = self.cap.__capture__()
        dataset, sv_img = self.pre.__preprocessor__()
        det = self.main.__mainprocessor__()

    def get_model_meta(self):
        model_meta={}

        model_meta['Object-confidence-threshold'] = self.conf_thres
        model_meta['IOU-threshold-for-NMS'] = self.iou_thres

        return model_meta
        

        
    
        