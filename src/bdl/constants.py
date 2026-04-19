# -*- coding: utf-8 -*-
"""
@author: PaRa
@author: igoforth
"""

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_TRAIN_SAMPLES = 25000
MAX_VALIDATION_SAMPLES = 5000
