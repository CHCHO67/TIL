import time

class MainProcessor():
  def __init__(self):
    self.model = None
    self.device = None

  def exec(self, batch):
    time.sleep(0.0015)
    return batch