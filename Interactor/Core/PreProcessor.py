import torch

class PreProcessor():
  def __init__(self):
    self.img_size = None

  def exec(self, frame):
    img = torch.from_numpy(frame)
    
    if img.ndimension() == 3:
      img = img.unsqeeze(0)

    return img