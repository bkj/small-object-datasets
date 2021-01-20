import numpy as np
from torch.utils.data import Dataset

import torch

class SODWrapper(Dataset):
  def __init__(self, dataset, obs_per_frame=4, frame_scale=4, seed=None):
    """
      obs_per_frame: number of objects per frame
      frame_scale: width / height of frame as multiple of width / height of input images
    """
    
    self.dataset       = dataset
    self.obs_per_frame = obs_per_frame
    self.frame_scale   = frame_scale
    self.seed          = seed
    
    self.nobs    = len(dataset)
    self.nobs    = self.nobs - self.nobs % obs_per_frame # round down
    self.nframes = self.nobs // obs_per_frame
    
    rng = np.random if seed is None else np.random.RandomState(seed=seed)
    self.idxs = rng.permutation(self.nobs).reshape(self.nframes, self.obs_per_frame)
  
  def __len__(self):
    return self.nframes
  
  def _try_sample(self, X, y, max_failures=10):
    """
      naive method for sampling non-overlapping boxes
      seems to work OK when the sampling doesn't have to be that dense 
    """
    c, h, w = X[0].shape
    
    Xf = torch.ones(c, self.frame_scale * h, self.frame_scale * w).float()
    yf = torch.zeros(1, self.frame_scale * h, self.frame_scale * w).long() - 1

    i = 0
    while True:
      h_off = rng.choice((self.frame_scale - 1) * h)
      w_off = rng.choice((self.frame_scale - 1) * w)
      
      no_overlap = (yf[:, h_off:h_off + h, w_off:w_off + w] == -1).all()
      if no_overlap:
        Xf[:, h_off:h_off + h, w_off:w_off + w] = X[i]
        yf[:, h_off:h_off + h, w_off:w_off + w] = y[i]
        i += 1
      else:
        failures += 1
      
      if i == len(X):
        return Xy, yf
      elif failures == max_failures:
        return None, None
    
  def __getitem__(self, frame_idx):
    rng = np.random if self.seed is None else np.random.RandomState(seed=self.seed + frame_idx)
    
    tmp  = [self.dataset[i] for i in self.idxs[frame_idx]]
    X, y = zip(*tmp)
    
    Xy, yf = None
    while Xf is None:
      Xy, yf = self._try_sample(X, y)
    
    return Xf, yf

