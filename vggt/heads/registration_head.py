import torch.nn as nn
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor
import matplotlib.pyplot as plt



class RegistrationHead(nn.Module):
    """
    RegistrationHead predicts homography matrix between two images.

    """