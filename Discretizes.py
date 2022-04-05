"""discretizes the observation for flappy bird"""
import numpy as np
from skimage import color

def discretizes(obs):
    return np.round(color.rgb2gray(obs)).astype(bool)[62:-8,:-110]

