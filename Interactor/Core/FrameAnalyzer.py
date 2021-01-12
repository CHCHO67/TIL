import cv2

from Interactor.Core.PreProcessor import PreProcessor
from Interactor.Core.MainProcessor import MainProcessor
from Interactor.Core.PostProcessor import PostProcessor
from Interactor.Core.drawBbox import drawBbox



class FrameAnalyzer():
    def __init__(self):
        self.preProcessor = PreProcessor()
        self.mainProcessor = MainProcessor()
        self.postProcessor = PostProcessor()
        self.drawbBox = drawBbox()
        
    def exec(self, frame):
        batch = self.preProcessor.exec(frame)
        # preProcessor return img
        pred = self.mainProcessor.exec(batch)
        # mainProcessor return batch
        # core mainProcessor has no-info
        bbox = self.postProcessor.exec(pred)
        # postProcessor return frame_box
        print(frame)
        drawBbox.exec(frame, bbox)
        #draw bbox with (frame, bbox)
        return bbox
        