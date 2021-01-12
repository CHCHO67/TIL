# Draw the predicted bounding box
import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#from Interactor.CocoDetector.AIAnalyzer import AIAnalyzer

from Adaptor.VideoCapturer import videoCapturer

class drawBbox():
  def __init__(self):
    self.classID = None
    self.left = None
    self.right = None
    self.top = None
    self.bottom = None
    self.conf = None
    self.frame = None

  # Draw the predicted bounding box
  #def exec(self, classID, conf, left, top, right, bottom):
  def exec(self, frame, bbox):
    print('drawing bounding box')
    #Draw a bounding box
    self.left = bbox.x_min
    self.right = bbox.x_max
    self.top = bbox.y_max
    self.bottom = bbox.y_min
    self.conf = bbox.score
    self.frame = frame

    cv2.rectangle(self.frame, (self.left, self.top), (self.right, self.bottom), (0,0,255))

    label = '%.2f' % self.conf

    # Get the label for the class name and its confidence
    classes = 1

    if classes:
      assert(classID < len(classes))
      label = '%s:%s' % (classes[self.classID], label)

    # Display the label at the top of the bounding box
    labelSize , baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(self.top, labelSize[1])
    cv2.rectangle(self.frame, (self.left, self.top - round(1.5*labelSize[1])), (self.left + round(1.5*labelSize[0]), self.top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(self.frame, label, (self.left, self.top), cv2.Font_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    #cv2.imwrite(outputFile, frame.astype(np.uint8))

