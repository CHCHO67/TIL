#!/usr/bin/env python3

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Adaptor.VideoCapturer import *
from Interactor.Core.FrameAnalyzer import FrameAnalyzer

class AIAnalyzer():
  def __init__(self, mode='video', path='', save_dir = None):
    if mode == 'video':
    from Adaptor.Capture import VideoCapture
    self.capture =VideoCapture(path)

    elif mode == 'image':
        from Adaptor.Capture import ImageCapture
        self.capture =ImageCapture(path)

    self.vid = vc.capture()
    self.frame_length = vc.get_frame_length()
    self.init_meta  = vc.get_video_meta()
    
    self.frameanalyzer = FrameAnalyzer()

  def exec(self):
    print('print video-information')
    print(self.init_meta)
    print('Start')

    #save frame in list
    frames = []

    #for k in range(self.frame_length):
    while self.vid.isOpened(): 
      ret, frame = self.vid.read()
      if ret == False:
        print("Done processing!!!")
        print("Output file is stored as ", )
        cv2.waitKey(3000)
        break
      
      #get bbox
      box = self.frameanalyzer.exec(frame)
      # return bbox
      print(frame)
      print(box)
      #drawBbox(box, frame)
      #여기에서 바운딩 박스 그리기

      frames.append(box)

      if cv2.waitKey(1) == ord('q'):
        print('Finish!!')
        return frames
        #break

      #print('Finish')

      #self.release()

      #return frames

    def release(self):
      self.vid.release()
