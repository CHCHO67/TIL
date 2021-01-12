
class PostProcessor():
  def __init__(self):
    pass

  def exec(self, pred):
    
    b,c,h,w = pred.shape

    x1 = int(100)
    y1 = int(100)
    x2 = x1 * 2
    y2 = y1 * 2

    frame_box = []
    box_ = {'x_min': int(x1), 'y_min': int(x2), 
            'x_max': int(x2), 'y_max':int(y2),
            'kind': 'Test', 'score': 0.99}
    
    frame_box.append(box_)

    return frame_box